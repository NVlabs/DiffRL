# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import urdfpy
import math
import numpy as np
import os
import torch
import random

import xml.etree.ElementTree as ET

import dflex as df


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

def urdf_add_collision(builder, link, collisions, shape_ke, shape_kd, shape_kf, shape_mu):
        
    # add geometry
    for collision in collisions:
        
        origin = urdfpy.matrix_to_xyz_rpy(collision.origin)

        pos = origin[0:3]
        rot = df.rpy2quat(*origin[3:6])

        geo = collision.geometry

        if (geo.box):
            builder.add_shape_box(
                link,
                pos, 
                rot, 
                geo.box.size[0]*0.5, 
                geo.box.size[1]*0.5, 
                geo.box.size[2]*0.5,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
        
        if (geo.sphere):
            builder.add_shape_sphere(
                link, 
                pos, 
                rot, 
                geo.sphere.radius,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
         
        if (geo.cylinder):
            
            # cylinders in URDF are aligned with z-axis, while dFlex uses x-axis
            r = df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.5)

            builder.add_shape_capsule(
                link, 
                pos, 
                df.quat_multiply(rot, r), 
                geo.cylinder.radius, 
                geo.cylinder.length*0.5,
                ke=shape_ke,
                kd=shape_kd,
                kf=shape_kf,
                mu=shape_mu)
 
        if (geo.mesh):

            for m in geo.mesh.meshes:
                faces = []
                vertices = []

                for v in m.vertices:
                    vertices.append(np.array(v))
                    
                for f in m.faces:
                    faces.append(int(f[0]))
                    faces.append(int(f[1]))
                    faces.append(int(f[2]))
                                    
                mesh = df.Mesh(vertices, faces)
                
                builder.add_shape_mesh(
                    link,
                    pos,
                    rot,
                    mesh,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu)
      
def urdf_load(
    builder, 
    filename, 
    xform, 
    floating=False, 
    armature=0.0, 
    shape_ke=1.e+4, 
    shape_kd=1.e+4, 
    shape_kf=1.e+2, 
    shape_mu=0.25,
    limit_ke=100.0,
    limit_kd=1.0):

    robot = urdfpy.URDF.load(filename)

    # maps from link name -> link index
    link_index = {}

    builder.add_articulation()

    # add base
    if (floating):
        root = builder.add_link(-1, df.transform_identity(), (0,0,0), df.JOINT_FREE)

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform[0][0]
        builder.joint_q[start + 1] = xform[0][1]
        builder.joint_q[start + 2] = xform[0][2]

        builder.joint_q[start + 3] = xform[1][0]
        builder.joint_q[start + 4] = xform[1][1]
        builder.joint_q[start + 5] = xform[1][2]
        builder.joint_q[start + 6] = xform[1][3]
    else:    
        root = builder.add_link(-1, xform, (0,0,0), df.JOINT_FIXED)

    urdf_add_collision(builder, root, robot.links[0].collisions, shape_ke, shape_kd, shape_kf, shape_mu)
    link_index[robot.links[0].name] = root

    # add children
    for joint in robot.joints:

        type = None
        axis = (0.0, 0.0, 0.0)

        if (joint.joint_type == "revolute" or joint.joint_type == "continuous"):
            type = df.JOINT_REVOLUTE
            axis = joint.axis
        if (joint.joint_type == "prismatic"):
            type = df.JOINT_PRISMATIC
            axis = joint.axis
        if (joint.joint_type == "fixed"):
            type = df.JOINT_FIXED
        if (joint.joint_type == "floating"):
            type = df.JOINT_FREE
        
        parent = -1

        if joint.parent in link_index:
            parent = link_index[joint.parent]

        origin = urdfpy.matrix_to_xyz_rpy(joint.origin)

        pos = origin[0:3]
        rot = df.rpy2quat(*origin[3:6])

        lower = -1.e+3
        upper = 1.e+3
        damping = 0.0

        # limits
        if (joint.limit):
            
            if (joint.limit.lower != None):
                lower = joint.limit.lower
            if (joint.limit.upper != None):
                upper = joint.limit.upper

        # damping
        if (joint.dynamics):
            if (joint.dynamics.damping):
                damping = joint.dynamics.damping
        # add link
        link = builder.add_link(
            parent=parent, 
            X_pj=df.transform(pos, rot), 
            axis=axis, 
            type=type,
            limit_lower=lower,
            limit_upper=upper,
            limit_ke=limit_ke,
            limit_kd=limit_kd,
            damping=damping)

        # add collisions
        urdf_add_collision(builder, link, robot.link_map[joint.child].collisions, shape_ke, shape_kd, shape_kf, shape_mu)

        # add ourselves to the index
        link_index[joint.child] = link

# build an articulated tree
def build_tree(
    builder, 
    angle,
    max_depth,    
    width=0.05,
    length=0.25,
    density=1000.0,
    joint_stiffness=0.0,
    joint_damping=0.0,
    shape_ke = 1.e+4,
    shape_kd = 1.e+3,
    shape_kf = 1.e+2,
    shape_mu = 0.5,
    floating=False):

    def build_recursive(parent, depth):

        if (depth >= max_depth):
            return

        X_pj = df.transform((length * 2.0, 0.0, 0.0), df.quat_from_axis_angle((0.0, 0.0, 1.0), angle))

        type = df.JOINT_REVOLUTE
        axis = (0.0, 0.0, 1.0)

        if (depth == 0 and floating == True):
            X_pj = df.transform((0.0, 0.0, 0.0), df.quat_identity())
            type = df.JOINT_FREE

        link = builder.add_link(
            parent, 
            X_pj, 
            axis, 
            type,
            stiffness=joint_stiffness,
            damping=joint_damping)
        
        # capsule
        shape = builder.add_shape_capsule(
            link, 
            pos=(length, 0.0, 0.0), 
            radius=width, 
            half_width=length, 
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
            mu=shape_mu)

        # recurse
        #build_tree_recursive(builder, link, angle, width, depth + 1, max_depth, shape_ke, shape_kd, shape_kf, shape_mu, floating)
        build_recursive(link, depth + 1)

    # 
    build_recursive(-1, 0)

# Mujoco file format parser

def parse_mjcf(
    filename, 
    builder, 
    density=1000.0, 
    stiffness=0.0, 
    damping=1.0, 
    contact_ke=1e4,
    contact_kd=1e4,
    contact_kf=1e3,
    contact_mu=0.5,
    limit_ke=100.0,
    limit_kd=10.0,
    armature=0.01,
    radians=False,
    load_stiffness=False,
    load_armature=False):

    file = ET.parse(filename)
    root = file.getroot()

    type_map = { 
        "ball": df.JOINT_BALL, 
        "hinge": df.JOINT_REVOLUTE, 
        "slide": df.JOINT_PRISMATIC, 
        "free": df.JOINT_FREE, 
        "fixed": df.JOINT_FIXED
    }
    
    def parse_float(node, key, default):
        if key in node.attrib:
            return float(node.attrib[key])
        else:
            return default

    def parse_bool(node, key, default):
        if key in node.attrib:
            
            if node.attrib[key] == "true":
                return True
            else:
                return False

        else:
            return default

    def parse_vec(node, key, default):
        if key in node.attrib:
            return np.fromstring(node.attrib[key], sep=" ")
        else:
            return np.array(default)

    def parse_body(body, parent, last_joint_pos):

        body_name = body.attrib["name"]
        body_pos = np.fromstring(body.attrib["pos"], sep=" ")
        # last_joint_pos = np.zeros(3)

        #-----------------
        # add body for each joint, we assume the joints attached to one body have the same joint_pos

        for i, joint in enumerate(body.findall("joint")):
            
            joint_name = joint.attrib["name"]
            joint_type = type_map[joint.attrib.get("type", 'hinge')]
            joint_axis = parse_vec(joint, "axis", (0.0, 0.0, 0.0))
            joint_pos = parse_vec(joint, "pos", (0.0, 0.0, 0.0))
            joint_limited = parse_bool(joint, "limited", True)
            if joint_limited:
                if radians:
                    joint_range = parse_vec(joint, "range", (np.deg2rad(-170.), np.deg2rad(170.)))
                else:
                    joint_range = np.deg2rad(parse_vec(joint, "range", (-170.0, 170.0)))
            else:
                joint_range = np.array([-1.e+6, 1.e+6])

            if load_stiffness:
                joint_stiffness = parse_float(joint, 'stiffness', stiffness)
            else:
                joint_stiffness = stiffness

            joint_damping = parse_float(joint, 'damping', damping)

            if load_armature:
                joint_armature = parse_float(joint, "armature", armature)
            else:
                joint_armature = armature

            joint_axis = df.normalize(joint_axis)
            
            if (parent == -1):
                body_pos = np.array((0.0, 0.0, 0.0))
            
            #-----------------
            # add body
            link = builder.add_link(
                parent, 
                X_pj=df.transform(body_pos + joint_pos - last_joint_pos, df.quat_identity()),
                axis=joint_axis, 
                type=joint_type,
                limit_lower=joint_range[0],
                limit_upper=joint_range[1],
                limit_ke=limit_ke,
                limit_kd=limit_kd,
                stiffness=joint_stiffness,
                damping=joint_damping,
                armature=joint_armature)

            # assume that each joint is one body in simulation
            parent = link               
            body_pos = [0.0, 0.0, 0.0]  
            last_joint_pos = joint_pos

        #-----------------
        # add shapes to the last joint in the body

        for geom in body.findall("geom"):
            geom_name = geom.attrib["name"]
            geom_type = geom.attrib["type"]

            geom_size = parse_vec(geom, "size", [1.0])                
            geom_pos = parse_vec(geom, "pos", (0.0, 0.0, 0.0)) 
            geom_rot = parse_vec(geom, "quat", (0.0, 0.0, 0.0, 1.0))

            if (geom_type == "sphere"):

                builder.add_shape_sphere(
                    link, 
                    pos=geom_pos - last_joint_pos, # position relative to the parent frame
                    rot=geom_rot,
                    radius=geom_size[0],
                    density=density,
                    ke=contact_ke,
                    kd=contact_kd,
                    kf=contact_kf,
                    mu=contact_mu)

            elif (geom_type == "capsule"):

                if ("fromto" in geom.attrib):
                    geom_fromto = parse_vec(geom, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

                    start = geom_fromto[0:3]
                    end = geom_fromto[3:6]

                    # compute rotation to align dflex capsule (along x-axis), with mjcf fromto direction                        
                    axis = df.normalize(end-start)
                    angle = math.acos(np.dot(axis, (1.0, 0.0, 0.0)))
                    axis = df.normalize(np.cross(axis, (1.0, 0.0, 0.0)))

                    geom_pos = (start + end)*0.5
                    geom_rot = df.quat_from_axis_angle(axis, -angle)

                    geom_radius = geom_size[0]
                    geom_width = np.linalg.norm(end-start)*0.5

                else:

                    geom_radius = geom_size[0]
                    geom_width = geom_size[1]
                    geom_pos = parse_vec(geom, "pos", (0.0, 0.0, 0.0))
                
                    if ("axisangle" in geom.attrib):
                        axis_angle = parse_vec(geom, "axisangle", (0.0, 1.0, 0.0, 0.0))
                        geom_rot = df.quat_from_axis_angle(axis_angle[0:3], axis_angle[3])

                    if ("quat" in geom.attrib):
                        q = parse_vec(geom, "quat", df.quat_identity())
                        geom_rot = q

                    geom_rot = df.quat_multiply(geom_rot, df.quat_from_axis_angle((0.0, 1.0, 0.0), -math.pi*0.5))

                builder.add_shape_capsule(
                    link,
                    pos=geom_pos - last_joint_pos,
                    rot=geom_rot,
                    radius=geom_radius,
                    half_width=geom_width,
                    density=density,
                    ke=contact_ke,
                    kd=contact_kd,
                    kf=contact_kf,
                    mu=contact_mu)

            else:
                print("Type: " + geom_type + " unsupported")        

        #-----------------
        # recurse

        for child in body.findall("body"):
            parse_body(child, link, last_joint_pos)

    #-----------------
    # start articulation

    builder.add_articulation()

    world = root.find("worldbody")
    for body in world.findall("body"):
        parse_body(body, -1, np.zeros(3))


# SNU file format parser

class MuscleUnit:

    def __init__(self):
        
        self.name = ""
        self.bones = []
        self.points = []
        self.muscle_strength = 0.0

class Skeleton:

    def __init__(self, skeleton_file, muscle_file, builder, 
        filter={}, 
        visualize_shapes=True, 
        stiffness=5.0, 
        damping=2.0, 
        contact_ke=5000.0,
        contact_kd=2000.0,
        contact_kf=1000.0,
        contact_mu=0.5,
        limit_ke=1000.0,
        limit_kd=10.0,
        armature = 0.05):

        self.armature = armature
        self.stiffness = stiffness
        self.damping = damping

        self.contact_ke = contact_ke
        self.contact_kd = contact_kd
        self.contact_kf = contact_kf

        self.limit_ke = limit_ke
        self.limit_kd = limit_kd

        self.contact_mu = contact_mu

        self.visualize_shapes = visualize_shapes

        self.parse_skeleton(skeleton_file, builder, filter)

        if muscle_file != None:
            self.parse_muscles(muscle_file, builder)

    def parse_skeleton(self, filename, builder, filter):
        file = ET.parse(filename)
        root = file.getroot()
        
        self.node_map = {}       # map node names to link indices
        self.xform_map = {}      # map node names to parent transforms
        self.mesh_map = {}       # map mesh names to link indices objects

        self.coord_start = len(builder.joint_q)
        self.dof_start = len(builder.joint_qd)

        type_map = { 
            "Ball": df.JOINT_BALL, 
            "Revolute": df.JOINT_REVOLUTE, 
            "Prismatic": df.JOINT_PRISMATIC, 
            "Free": df.JOINT_FREE, 
            "Fixed": df.JOINT_FIXED
        }

        builder.add_articulation()

        for child in root:

            if (child.tag == "Node"):

                body = child.find("Body")
                joint = child.find("Joint")

                name = child.attrib["name"]
                parent = child.attrib["parent"]
                parent_X_s = df.transform_identity()

                if parent in self.node_map:
                    parent_link = self.node_map[parent]
                    parent_X_s = self.xform_map[parent]
                else:
                    parent_link = -1

                body_xform = body.find("Transformation")
                joint_xform = joint.find("Transformation")

                body_mesh = body.attrib["obj"]
                body_size = np.fromstring(body.attrib["size"], sep=" ")
                body_type = body.attrib["type"]
                body_mass = float(body.attrib["mass"])

                x=body_size[0]
                y=body_size[1]
                z=body_size[2]
                density = body_mass / (x*y*z)

                max_body_mass = 15.0
                mass_scale = body_mass / max_body_mass

                body_R_s = np.fromstring(body_xform.attrib["linear"], sep=" ").reshape((3,3))
                body_t_s = np.fromstring(body_xform.attrib["translation"], sep=" ")

                joint_R_s = np.fromstring(joint_xform.attrib["linear"], sep=" ").reshape((3,3))
                joint_t_s = np.fromstring(joint_xform.attrib["translation"], sep=" ")
            
                joint_type = type_map[joint.attrib["type"]]

                joint_lower = -1.e+3
                joint_upper = 1.e+3
                
                if (joint_type == type_map["Revolute"]):
                    if ("lower" in joint.attrib):
                        joint_lower = np.fromstring(joint.attrib["lower"], sep=" ")[0]

                    if ("upper" in joint.attrib):  
                        joint_upper = np.fromstring(joint.attrib["upper"], sep=" ")[0]
                
                    # print(joint_type, joint_lower, joint_upper)

                if ("axis" in joint.attrib):
                    joint_axis = np.fromstring(joint.attrib["axis"], sep=" ")
                else:
                    joint_axis = np.array((0.0, 0.0, 0.0))

                body_X_s = df.transform(body_t_s, df.quat_from_matrix(body_R_s))
                joint_X_s = df.transform(joint_t_s, df.quat_from_matrix(joint_R_s))

                mesh_base = os.path.splitext(body_mesh)[0]
                mesh_file = mesh_base + ".usd"

                link = -1

                if len(filter) == 0 or name in filter:

                    joint_X_p = df.transform_multiply(df.transform_inverse(parent_X_s), joint_X_s)
                    body_X_c = df.transform_multiply(df.transform_inverse(joint_X_s), body_X_s)

                    if (parent_link == -1):
                        joint_X_p = df.transform_identity()

                    # add link
                    link = builder.add_link(
                        parent=parent_link, 
                        X_pj=joint_X_p,
                        axis=joint_axis,
                        type=joint_type,
                        limit_lower=joint_lower,
                        limit_upper=joint_upper,
                        limit_ke=self.limit_ke * mass_scale,
                        limit_kd=self.limit_kd * mass_scale,
                        damping=self.damping,
                        stiffness=self.stiffness * math.sqrt(mass_scale),
                        armature=self.armature)
                        # armature=self.armature * math.sqrt(mass_scale)) 

                    # add shape
                    shape = builder.add_shape_box(
                        body=link, 
                        pos=body_X_c[0],
                        rot=body_X_c[1],
                        hx=x*0.5,
                        hy=y*0.5,
                        hz=z*0.5,
                        density=density,
                        ke=self.contact_ke,
                        kd=self.contact_kd,
                        kf=self.contact_kf,
                        mu=self.contact_mu)

                # add lookup in name->link map
                # save parent transform
                self.xform_map[name] = joint_X_s
                self.node_map[name] = link
                self.mesh_map[mesh_base] = link

    def parse_muscles(self, filename, builder):

        # list of MuscleUnits
        muscles = []

        file = ET.parse(filename)
        root = file.getroot()

        self.muscle_start = len(builder.muscle_activation)

        for child in root:

            if (child.tag == "Unit"):

                unit_name = child.attrib["name"]
                unit_f0 = float(child.attrib["f0"])
                unit_lm = float(child.attrib["lm"])
                unit_lt = float(child.attrib["lt"])
                unit_lmax = float(child.attrib["lmax"])
                unit_pen = float(child.attrib["pen_angle"])

                m = MuscleUnit()
                m.name = unit_name

                m.muscle_strength = unit_f0

                incomplete = False

                for waypoint in child.iter("Waypoint"):
                
                    way_bone = waypoint.attrib["body"]
                    way_link = self.node_map[way_bone]
                    way_loc = np.fromstring(waypoint.attrib["p"], sep=" ", dtype=np.float32)

                    if (way_link == -1):
                        incomplete = True
                        break

                    # transform loc to joint local space
                    joint_X_s = self.xform_map[way_bone]

                    way_loc = df.transform_point(df.transform_inverse(joint_X_s), way_loc)

                    m.bones.append(way_link)
                    m.points.append(way_loc)

                if not incomplete:

                    muscles.append(m)
                    builder.add_muscle(m.bones, m.points, f0=unit_f0, lm=unit_lm, lt=unit_lt, lmax=unit_lmax, pen=unit_pen)

        self.muscles = muscles



