# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
from utils.common import *

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.
    
    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(self.name)
        self.start_time = time.time()
        
    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(self.name)
        self.time_total += time.time() - self.start_time
        self.start_time = None
    
    def report(self):
        print_info('Time report [{}]: {:.2f} seconds'.format(self.name, self.time_total))

    def clear(self):
        self.start_time = None
        self.time_total = 0.

class TimeReport:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name = name)
    
    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()
    
    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()
    
    def report(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")
            for timer_name in self.timers.keys():
                self.timers[timer_name].report()
            print_info("-----------------------------------")
    
    def clear_timer(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()
    
    def pop_timer(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}
