# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""dFlex Kit extension

Allows setting up, training, and running inference on dFlex optimization environments.

"""

import os
import subprocess
import carb
import carb.input
import math
import numpy as np
import omni.kit.ui
import omni.appwindow
import omni.kit.editor
import omni.timeline
import omni.usd
import omni.ui as ui

from pathlib import Path

ICON_PATH = Path(__file__).parent.parent.joinpath("icons")


from pxr import Usd, UsdGeom, Sdf, Gf

import torch

from omni.kit.settings import create_setting_widget, create_setting_widget_combo, SettingType, get_settings_interface

KIT_GREEN = 0xFF8A8777
LABEL_PADDING = 120

DARK_WINDOW_STYLE = {
    "Button": {"background_color": 0xFF292929, "margin": 3, "padding": 3, "border_radius": 2},
    "Button.Label": {"color": 0xFFCCCCCC},
    "Button:hovered": {"background_color": 0xFF9E9E9E},
    "Button:pressed": {"background_color": 0xC22A8778},
    "VStack::main_v_stack": {"secondary_color": 0x0, "margin_width": 10, "margin_height": 0},
    "VStack::frame_v_stack": {"margin_width": 15, "margin_height": 10},
    "Rectangle::frame_background": {"background_color": 0xFF343432, "border_radius": 5},
    "Field::models": {"background_color": 0xFF23211F, "font_size": 14, "color": 0xFFAAAAAA, "border_radius": 4.0},
    "Frame": {"background_color": 0xFFAAAAAA},
    "Label": {"font_size": 14, "color": 0xFF8A8777},
    "Label::status": {"font_size": 14, "color": 0xFF8AFF77}
}

CollapsableFrame_style = {
    "CollapsableFrame": {
        "background_color": 0xFF343432,
        "secondary_color": 0xFF343432,
        "color": 0xFFAAAAAA,
        "border_radius": 4.0,
        "border_color": 0x0,
        "border_width": 0,
        "font_size": 14,
        "padding": 0,
    },
    "HStack::header": {"margin": 5},
    "CollapsableFrame:hovered": {"secondary_color": 0xFF3A3A3A},
    "CollapsableFrame:pressed": {"secondary_color": 0xFF343432},
}


experiment = None


class Extension:

    def __init__(self):
        self.MENU_SHOW_WINDOW = "Window/dFlex"
        self.MENU_INSERT_REFERENCE = "Utilities/Insert Reference"

        self._editor_window = None
        self._window_Frame = None

        self.time = 0.0
        
        self.plot = None
        self.log = None
        self.status = None

        self.mode = 'stopped'

        self.properties = {}

        # add some helper menus
        self.menus = []

    def on_shutdown(self):

        self._editor_window = None

        self.menus = []
        #self.input.unsubscribe_to_keyboard_events(self.appwindow.get_keyboard(), self.key_sub)


    def on_startup(self):
        self.appwindow = omni.appwindow.get_default_app_window()
        self.editor = omni.kit.editor.get_editor_interface()
        self.input = carb.input.acquire_input_interface()
        self.timeline = omni.timeline.get_timeline_interface()
        self.usd_context = omni.usd.get_context()

        # event subscriptions
        self.stage_sub = self.usd_context.get_stage_event_stream().create_subscription_to_pop(self.on_stage, name="dFlex")
        self.update_sub = self.editor.subscribe_to_update_events(self.on_update)
        #self.key_sub = self.input.subscribe_to_keyboard_events(self.appwindow.get_keyboard(), self.on_key)

        self.menus.append(omni.kit.ui.get_editor_menu().add_item(self.MENU_SHOW_WINDOW, self.ui_on_menu, True, 11))
        self.menus.append(omni.kit.ui.get_editor_menu().add_item(self.MENU_INSERT_REFERENCE, self.ui_on_menu))

        self.reload()
        self.build_ui()

    def format(self, s):
        return s.replace("_", " ").title()

    def add_float_field(self, label, x, low=0.0, high=1.0):
        with ui.HStack():
            ui.Label(self.format(label), width=120)
            self.add_property(label, ui.FloatSlider(name="value", width=150, min=low, max=high), x)

    def add_int_field(self, label, x, low=0, high=100):
        with ui.HStack():
            ui.Label(self.format(label), width=120)
            self.add_property(label, ui.IntSlider(name="value", width=150, min=low, max=high), x)

    def add_combo_field(self, label, i, options):
        with ui.HStack():
            ui.Label(self.format(label), width=120)
            ui.ComboBox(i, *options, width=150) # todo: how does the model work for combo boxes in omni.ui

    def add_bool_field(self, label, b):
        with ui.HStack():
            ui.Label(self.format(label), width=120)
            self.add_property(label, ui.CheckBox(width=10), b)

    def add_property(self, label, widget, value):
        self.properties[label] = widget
        widget.model.set_value(value)

    def ui_on_menu(self, menu, value):
        if menu == self.MENU_SHOW_WINDOW:

            if self.window:
                if value:
                    self.window.show()
                else:
                    self.window.hide()

            omni.kit.ui.get_editor_menu().set_value(self.STAGE_SCRIPT_WINDOW_MENU, value)

        if menu == self.MENU_INSERT_REFERENCE:
            self.file_pick = omni.kit.ui.FilePicker("Select USD File", file_type=omni.kit.ui.FileDialogSelectType.FILE)
            self.file_pick.set_file_selected_fn(self.ui_on_select_ref_fn)
            self.file_pick.show(omni.kit.ui.FileDialogDataSource.LOCAL)

    def ui_on_select_ref_fn(self, real_path):

        file = os.path.normpath(real_path)
        name = os.path.basename(file)
        stem = os.path.splitext(name)[0]

        stage = self.usd_context.get_stage()
        stage_path = stage.GetRootLayer().realPath

        base = os.path.commonpath([real_path, stage_path]) 
        rel_path = os.path.relpath(real_path, base)

        over = stage.OverridePrim('/' + stem)
        over.GetReferences().AddReference(rel_path)


    def ui_on_select_script_fn(self):

        # file picker
        self.file_pick = omni.kit.ui.FilePicker("Select Python Script", file_type=omni.kit.ui.FileDialogSelectType.FILE)
        self.file_pick.set_file_selected_fn(self.set_stage_script)
        self.file_pick.add_filter("Python Files (*.py)", ".*.py")

        self.file_pick.show(omni.kit.ui.FileDialogDataSource.LOCAL)

    def ui_on_clear_script_fn(self, widget):
        self.clear_stage_script()


    def ui_on_select_network_fn(self):

        # file picker
        self.file_pick = omni.kit.ui.FilePicker("Select Model", file_type=omni.kit.ui.FileDialogSelectType.FILE)
        self.file_pick.set_file_selected_fn(self.set_network)
        self.file_pick.add_filter("PyTorch Files (*.pt)", ".*.pt")

        self.file_pick.show(omni.kit.ui.FileDialogDataSource.LOCAL)


    # build panel
    def build_ui(self):

        stage = self.usd_context.get_stage()

        self._editor_window = ui.Window("dFlex", width=450, height=800)
        self._editor_window.frame.set_style(DARK_WINDOW_STYLE)

        with self._editor_window.frame:
            with ui.VStack():
                self._window_Frame = ui.ScrollingFrame(
                    name="canvas",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                )            

                with self._window_Frame:                    
                    with ui.VStack(spacing=6, name="main_v_stack"):
                        ui.Spacer(height=5)
                        with ui.CollapsableFrame(title="Experiment", height=60, style=CollapsableFrame_style):
                            with ui.VStack(spacing=4, name="frame_v_stack"):
                                with ui.HStack():
                                    ui.Label("Script", name="label", width=120)

                                    s = ""
                                    if (self.get_stage_script() != None):
                                        s = self.get_stage_script()

                                    ui.StringField(name="models", tooltip="Training Python script").model.set_value(self.get_stage_script())
                                    ui.Button("", image_url="resources/icons/folder.png", width=15, image_width=15, clicked_fn=self.ui_on_select_script_fn)
                                    ui.Button("Clear", width=15, clicked_fn=self.clear_stage_script)
                                    ui.Button("Reload", width=15, clicked_fn=self.reload)

                                with ui.HStack():
                                    ui.Label("Hot Reload", width=100)
                                    ui.CheckBox(width=10).model.set_value(False)

                        if (experiment):

                            with ui.CollapsableFrame(height=60, title="Simulation Settings", style=CollapsableFrame_style):
                                with ui.VStack(spacing=4, name="frame_v_stack"):
                                    self.add_int_field("sim_substeps", 4, 1, 100)
                                    self.add_float_field("sim_duration", 5.0, 0.0, 30.0)


                            with ui.CollapsableFrame(title="Training Settings", height=60, style=CollapsableFrame_style):
                                with ui.VStack(spacing=4, name="frame_v_stack"):
                                    self.add_int_field("train_iters", 64, 1, 100)
                                    self.add_float_field("train_rate", 0.1, 0.0, 10.0)
                                    self.add_combo_field("train_optimizer", 0, ["GD", "SGD", "L-BFGS"])


                            with ui.CollapsableFrame(title="Actions", height=10, style=CollapsableFrame_style):
                                with ui.VStack(spacing=4, name="frame_v_stack"):
                                    with ui.HStack():
                                        ui.Label("Network", name="label", width=120)
                                        
                                        s = ""
                                        if (self.get_network() != None):
                                            s = self.get_network()

                                        ui.StringField(name="models", tooltip="Pretrained PyTorch network").model.set_value(s)
                                        ui.Button("", image_url="resources/icons/folder.png", width=15, image_width=15, clicked_fn=self.ui_on_select_network_fn)
                                        ui.Button("Clear", width=15, clicked_fn=self.clear_network)

                                    with ui.HStack():
                                        p = (1.0/6.0)*100.0
                                        ui.Button("Run", width=ui.Percent(p), clicked_fn=self.run)
                                        ui.Button("Train", width=ui.Percent(p), clicked_fn=self.train)
                                        ui.Button("Stop", width=ui.Percent(p), clicked_fn=self.stop)
                                        ui.Button("Reset", width=ui.Percent(p), clicked_fn=self.reset)

                                    self.add_bool_field("record", True)

                                    with ui.HStack():
                                        ui.Label("Status: ", width=120)
                                        self.status = ui.Label("", name="status", width=200)
                                        


                            with ui.CollapsableFrame(title="Loss", style=CollapsableFrame_style):
                                data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
                                self.plot = ui.Plot(ui.Type.LINE, -1.0, 1.0, *data, height=200, style={"color": 0xff00ffFF})

                            # with ui.ScrollingFrame(
                            #     name="log",
                            #     horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                            #     height=200,
                            #     width=ui.Percent(95)
                            # ):
                            with ui.CollapsableFrame(title="Log", style=CollapsableFrame_style):
                                with ui.VStack(spacing=4, name="frame_v_stack"):
                                    self.log = ui.Label("", height=200)

   

    def reload(self):

        path = self.get_stage_script()

        if (path):
            # read code to string
            file = open(path)
            code = file.read()
            file.close()

            # run it in the local environment
            exec(code, globals(), globals())

            self.build_ui()

    # methods for storing script in stage metadata
    def get_stage_script(self):

        stage = self.usd_context.get_stage()
        custom_data = stage.GetEditTarget().GetLayer().customLayerData

        print(custom_data)

        if "script" in custom_data:
            return custom_data["script"]
        else:
            return None


    def set_stage_script(self, real_path):

        path = os.path.normpath(real_path)
        
        print("Setting stage script to: " + str(path))
        stage = self.usd_context.get_stage()

        with Sdf.ChangeBlock():
            custom_data = stage.GetEditTarget().GetLayer().customLayerData
            custom_data["script"] = path

            stage.GetEditTarget().GetLayer().customLayerData = custom_data

        # rebuild ui
        self.build_ui()

  
    def clear_stage_script(self):
       
        stage = self.usd_context.get_stage()

        with Sdf.ChangeBlock():
            custom_data = stage.GetEditTarget().GetLayer().customLayerData

            if "script" in custom_data:
                del custom_data["script"]
                stage.GetEditTarget().GetLayer().customLayerData = custom_data

        self.build_ui()


    def set_network(self, real_path):

        path = os.path.normpath(real_path)
        experiment.network_file = path

        self.build_ui()

    def get_network(self):
        return experiment.network_file

    def clear_network(self):
        experiment.network_file = None

        self.build_ui()


    def on_key(self, event, *args, **kwargs):
        # if event.keyboard == self.appwindow.get_keyboard():
        #     if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        #         if event.input == carb.input.KeyboardInput.ESCAPE:
        #             self.stop()

        # return True
        pass


    def on_stage(self, stage_event):
        if stage_event.type == int(omni.usd.StageEventType.OPENED):
            self.build_ui()
            self.reload()

               
    def on_update(self, dt):

        if (experiment):

            stage = self.usd_context.get_stage()

            stage.SetStartTimeCode(0.0)
            stage.SetEndTimeCode(experiment.render_time*60.0)
            stage.SetTimeCodesPerSecond(60.0)

            # pass parameters to the experiment
            if ('record' in self.properties):
                experiment.record = self.properties['record'].model.get_value_as_bool()

            # experiment.train_rate = self.get_property('train_rate')
            # experiment.train_iters = self.get_property('train_iters')
            # experiment.sim_duration = self.get_property('sim_duration')
            # experiment.sim_substeps = self.get_property('sim_substeps')

            if (self.mode == 'training'):
                experiment.train()

                # update error plot
                if (self.plot):
                    self.plot.scale_min = np.min(experiment.train_loss)
                    self.plot.scale_max = np.max(experiment.train_loss)
                    self.plot.set_data(*experiment.train_loss)

            elif (self.mode == 'inference'):
                experiment.run()

            # update stage time (allow scrubbing while stopped)
            if (self.mode != 'stopped'):
                self.timeline.set_current_time(experiment.render_time*60.0)

            # update log
            if (self.log):
                self.log.text = df.util.log_output


    def set_status(self, str):
        self.status.text = str

    def train(self):
        experiment.reset()
        self.mode = 'training'

        # update status
        self.set_status('Training in progress, press [ESC] to cancel')

    def run(self):
        experiment.reset()
        self.mode = 'inference'

        # update status
        self.set_status('Inference in progress, press [ESC] to cancel')

    def stop(self):
        self.mode = 'stopped'

        # update status
        self.set_status('Stopped')

    def reset(self):
        experiment.reset()   
        self.stop()

def get_extension():
    return Extension()
