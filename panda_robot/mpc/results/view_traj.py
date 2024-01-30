from os.path import dirname, join, abspath

import pinocchio as pin
import hppfcl
import numpy as np
np.set_printoptions(precision=4, linewidth=180)

from wrapper_meshcat import MeshcatWrapper
from utils import RED, RED_FULL, GREEN, GREEN_FULL, BLACK, BLACK_FULL, BLUE, BLUE_FULL

from Result import Result

RED = np.array([249, 136, 126, 125]) / 255


def load_pinocchio_robot_panda(obstacle_dim = 0.1, obstacle_pose = pin.SE3.Identity()):
    """Load the robot from the models folder.

    Returns:
        rmodel, vmodel, cmodel: Robot model, visual model & collision model of the robot.
    """

    ### LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(
        dirname(dirname(dirname(dirname(str(abspath(__file__))))))), "models"
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
    
    for obj in cmodel.geometryObjects:
        if isinstance(obj.geometry, hppfcl.Sphere) or isinstance(obj.geometry, hppfcl.Cylinder) : 
            obj.meshColor = RED_FULL
        if isinstance(obj.geometry, hppfcl.Box) and not "finger" in obj.name and not "camera" in obj.name:
            obj.meshColor = BLACK
    return rmodel, vmodel, cmodel

rmodel,vmodel,cmodel = load_pinocchio_robot_panda()

TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, -0.4, 1.5]))
TARGET_POSE2 = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, -0.0, 1.5]))


MeshcatVis = MeshcatWrapper()
vis, meshcatVis = MeshcatVis.visualize(
    TARGET_POSE,
    OBSTACLE_DIM=1e-1,
    OBSTACLE=pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, -0.2, 1.5])),
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
)
MeshcatVis._renderSphere("target2", 0.05, TARGET_POSE2, BLUE)

INITIAL_CONFIG = pin.neutral(rmodel)
INITIAL_CONFIG = np.array([0.1, 0.7, 0.0, 0.7, -0.5, 1.5, 0.0])

vis.display(INITIAL_CONFIG)


curr_path = dirname(str(abspath(__file__)))
test = Result(curr_path + "/scene1/comparingcsqpfddpsqp/100_sqp .json")
test._results["weights"]
print(test._results["weights"])

print(test.get_Q().shape)

for q in test.get_Q():
    vis.display(q)