'''
Example script : MPC simulation with KUKA arm 
static target reaching task
'''
from os.path import dirname, join, abspath

import pinocchio as pin
import hppfcl
import numpy as np
np.set_printoptions(precision=4, linewidth=180)

import pin_utils, mpc_utils
from mim_robots.pybullet.wrapper import PinBulletWrapper

RED = np.array([249, 136, 126, 125]) / 255
import pybullet

def load_pinocchio_robot_panda(scene, capsule = False, obstacle_shape = "sphere", obstacle_dim = 0.1, obstacle_pose = pin.SE3.Identity()):
    """Load the robot from the models folder.

    Returns:
        rmodel, vmodel, cmodel: Robot model, visual model & collision model of the robot.
    """

    ### LOADING THE ROBOT
    pinocchio_model_dir = join(
        dirname(dirname(str(abspath(__file__)))), "models"
    )
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)

    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    )
    rmodel, [vmodel, cmodel] = robot.model, [robot.visual_model, robot.collision_model]
    q0 = pin.neutral(rmodel)

    rmodel, [vmodel, cmodel] = pin.buildReducedModel(
        rmodel, [vmodel, cmodel], [1, 9, 10], q0
    )
    cmodel_copy = cmodel.copy()
    list_names_capsules = []
    if capsule:
        for geometry_object in cmodel_copy.geometryObjects:
            if isinstance(geometry_object.geometry, hppfcl.Sphere):
                cmodel.removeGeometryObject(geometry_object.name)
            # Only selecting the cylinders
            if isinstance(geometry_object.geometry, hppfcl.Cylinder):
                if (geometry_object.name[:-4] + "capsule") in list_names_capsules:
                    capsule = pin.GeometryObject(
                    geometry_object.name[:-4] + "capsule" + "1",
                    geometry_object.parentJoint,
                    geometry_object.parentFrame,
                    geometry_object.placement,
                    hppfcl.Capsule(geometry_object.geometry.radius, geometry_object.geometry.halfLength),
                    )
                    capsule.meshColor = RED
                    cmodel.addGeometryObject(capsule)
                    cmodel.removeGeometryObject(geometry_object.name)
                    list_names_capsules.append(geometry_object.name[:-4] + "capsule" + "1" )
                else:
                    capsule = pin.GeometryObject(
                    geometry_object.name[:-4] + "capsule",
                    geometry_object.parentJoint,
                    geometry_object.parentFrame,
                    geometry_object.placement,
                    hppfcl.Capsule(geometry_object.geometry.radius, geometry_object.geometry.halfLength),
                    )
                    capsule.meshColor = RED
                    cmodel.addGeometryObject(capsule)
                    cmodel.removeGeometryObject(geometry_object.name)
                    list_names_capsules.append(geometry_object.name[:-4] + "capsule")

    if scene == 1:
        ### CREATING THE SPHERE ON THE UNIVERSE
        OBSTACLE_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, -0.2, 1.5]))
        OBSTACLE = hppfcl.Sphere(1e-1)
        OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
            "obstacle",
            rmodel.getFrameId("universe"),
            rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
            OBSTACLE,
            OBSTACLE_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)
    elif scene ==2:
        OBSTACLE_RADIUS = 1.0e-1
        OBSTACLE_HALFLENGTH = 0.25e-0
        OBSTACLE_HEIGHT = 0.85
        OBSTACLE1_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2), np.array([-0.0, 0.0, OBSTACLE_HEIGHT]))
        OBSTACLE1 = hppfcl.Capsule(OBSTACLE_RADIUS, OBSTACLE_HALFLENGTH)
        OBSTACLE1_GEOM_OBJECT = pin.GeometryObject(
        "obstacle1",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE1,
        OBSTACLE1_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE1_GEOM_OBJECT)
        OBSTACLE2_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2), np.array([-0.0, 0.4, OBSTACLE_HEIGHT]))
        OBSTACLE2 = hppfcl.Capsule(OBSTACLE_RADIUS, OBSTACLE_HALFLENGTH)
        OBSTACLE2_GEOM_OBJECT = pin.GeometryObject(
        "obstacle2",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE2,
        OBSTACLE2_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE2_GEOM_OBJECT)
        
        OBSTACLE3_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2) @ pin.utils.rotate("x", np.pi/2) , np.array([0.25, 0.2, OBSTACLE_HEIGHT]))
        OBSTACLE3 = hppfcl.Capsule(OBSTACLE_RADIUS, OBSTACLE_HALFLENGTH)
        OBSTACLE3_GEOM_OBJECT = pin.GeometryObject(
        "obstacle3",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE3,
        OBSTACLE3_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE3_GEOM_OBJECT)
        
        OBSTACLE4_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2) @ pin.utils.rotate("x", np.pi/2) , np.array([-0.25, 0.2, OBSTACLE_HEIGHT]))
        OBSTACLE4 = hppfcl.Capsule(OBSTACLE_RADIUS, OBSTACLE_HALFLENGTH)
        OBSTACLE4_GEOM_OBJECT = pin.GeometryObject(
        "obstacle4",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE4,
        OBSTACLE4_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE4_GEOM_OBJECT)
        
    elif scene ==3: 
        OBSTACLE_HEIGHT = 0.85 
        OBSTACLE_X = 2.0e-1
        OBSTACLE_Y = 0.5e-2
        OBSTACLE_Z = 0.5
        OBSTACLE1_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2), np.array([-0.0, 0.0, OBSTACLE_HEIGHT]))
        OBSTACLE1 = hppfcl.Box(OBSTACLE_X, OBSTACLE_Y,OBSTACLE_Z)
        OBSTACLE1_GEOM_OBJECT = pin.GeometryObject(
        "obstacle1",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE1,
        OBSTACLE1_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE1_GEOM_OBJECT)
        OBSTACLE2_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2), np.array([-0.0, 0.45, OBSTACLE_HEIGHT]))
        OBSTACLE2 = hppfcl.Box(OBSTACLE_X, OBSTACLE_Y,OBSTACLE_Z)
        OBSTACLE2_GEOM_OBJECT = pin.GeometryObject(
        "obstacle2",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE2,
        OBSTACLE2_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE2_GEOM_OBJECT)
        
        OBSTACLE3_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2) @ pin.utils.rotate("x", np.pi/2) , np.array([0.25, 0.225, OBSTACLE_HEIGHT]))
        OBSTACLE3 = hppfcl.Box(OBSTACLE_X, OBSTACLE_Y,OBSTACLE_Z)
        OBSTACLE3_GEOM_OBJECT = pin.GeometryObject(
        "obstacle3",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE3,
        OBSTACLE3_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE3_GEOM_OBJECT)
        
        OBSTACLE4_POSE = pin.SE3(pin.utils.rotate("y", np.pi/2) @ pin.utils.rotate("x", np.pi/2) , np.array([-0.25, 0.225, OBSTACLE_HEIGHT]))
        OBSTACLE4 = hppfcl.Box(OBSTACLE_X, OBSTACLE_Y,OBSTACLE_Z)
        OBSTACLE4_GEOM_OBJECT = pin.GeometryObject(
        "obstacle4",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE4,
        OBSTACLE4_POSE,
        )
        ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE4_GEOM_OBJECT)
    else: 
        if "sphere" == obstacle_shape:
            OBSTACLE = hppfcl.Sphere(obstacle_dim)
            OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
                "obstacle",
                rmodel.getFrameId("universe"),
                rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
                OBSTACLE,
                obstacle_pose,
            )
            ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)
        elif "box" == obstacle_shape:
            OBSTACLE = hppfcl.Box(obstacle_dim)
            OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
                "obstacle",
                rmodel.getFrameId("universe"),
                rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
                OBSTACLE,
                obstacle_pose,
            )
            ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)
        elif "capsule" == obstacle_shape:
            OBSTACLE = hppfcl.Capsule(obstacle_dim)
            OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
                "obstacle",
                rmodel.getFrameId("universe"),
                rmodel.frames[rmodel.getFrameId("universe")].parentJoint,
                OBSTACLE,
                obstacle_pose,
            )
            ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)            

    robot_reduced = pin.robot_wrapper.RobotWrapper(rmodel, cmodel, vmodel)  

    return robot_reduced


class PandaRobot(PinBulletWrapper):
    '''
    Pinocchio-PyBullet wrapper class for the KUKA LWR iiwa 
    '''
    def __init__(self, qref=np.zeros(7), pos=None, orn=None, scene = 1, obstacle_shape = "sphere", obstacle_dim = 1e-1, obstacle_pose = pin.SE3.Identity()): 

        # Load the robot
        if pos is None:
            pos = [0.0, 0, 0.0]
        if orn is None:
            orn = pybullet.getQuaternionFromEuler([0, 0, 0])

        pinocchio_model_dir = join(
            dirname(dirname(str(abspath(__file__)))), "models"
        )
        print(pinocchio_model_dir)
        model_path = join(pinocchio_model_dir, "franka_description/robots")
        mesh_dir = pinocchio_model_dir
        urdf_filename = "franka2.urdf"
        urdf_model_path = join(join(model_path, "panda"), urdf_filename)

        self.urdf_path = urdf_model_path
        self.robotId = pybullet.loadURDF(
            self.urdf_path,
            pos, orn,
            # flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase=True)
        pybullet.getBasePositionAndOrientation(self.robotId)
        
        # Create the robot wrapper in pinocchio.
        robot_full = load_pinocchio_robot_panda(capsule=False, scene=scene, obstacle_shape=obstacle_shape, obstacle_dim=obstacle_dim, obstacle_pose=obstacle_pose)  

        
        # Query all the joints.
        num_joints = pybullet.getNumJoints(self.robotId)

        for ji in range(num_joints):
            pybullet.changeDynamics(self.robotId, 
                                    ji, 
                                    linearDamping=.04,
                                    angularDamping=0.04, 
                                    restitution=0.0, 
                                    lateralFriction=0.5)
          

        self.pin_robot = robot_full
        controlled_joints_names = ["panda2_joint1", "panda2_joint2", "panda2_joint3", "panda2_joint4", 
                                   "panda2_joint5", "panda2_joint6", "panda2_joint7"]
        
        self.base_link_name = "support_joint"
        self.end_eff_ids = []
        self.end_eff_ids.append(self.pin_robot.model.getFrameId('panda2_rightfinger'))
        self.nb_ee = len(self.end_eff_ids)
        self.joint_names = controlled_joints_names

        # Creates the wrapper by calling the super.__init__.          
        super(PandaRobot, self).__init__(
            self.robotId, 
            self.pin_robot,
            controlled_joints_names,
            ['panda2_finger_joint1'],
            useFixedBase=True)
        self.nb_dof = self.nv
        
    def forward_robot(self, q=None, dq=None):
        if q is None:
            q, dq = self.get_state()
        elif dq is None:
            raise ValueError("Need to provide q and dq or non of them.")

        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def start_recording(self, file_name):
        self.file_name = file_name
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def stop_recording(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)
        
        
