from os.path import dirname, join, abspath

import pinocchio as pin
import hppfcl
import numpy as np
np.set_printoptions(precision=4, linewidth=180)

from wrapper_meshcat import MeshcatWrapper

RED = np.array([249, 136, 126, 125]) / 255

### LOADING THE ROBOT
pinocchio_model_dir = join(
    dirname(dirname(dirname(dirname(str(abspath(__file__)))))), "models"
)
model_path = join(pinocchio_model_dir, "franka_description/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "franka3.urdf"
urdf_model_path = join(join(model_path, "panda"), urdf_filename)

robot = pin.RobotWrapper.BuildFromURDF(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)
rmodel, [vmodel, cmodel] = robot.model, [robot.visual_model, robot.collision_model]
q0 = pin.neutral(rmodel)

rmodel, [vmodel, cmodel] = pin.buildReducedModel(
    rmodel, [vmodel, cmodel], [1, 9, 10], q0
)

TARGET_POSE = pin.SE3.Identity()

MeshcatVis = MeshcatWrapper()
vis, meshcatVis = MeshcatVis.visualize(
    TARGET_POSE,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
)

INITIAL_CONFIG = pin.neutral(rmodel)
INITIAL_CONFIG = np.array([0.5,0.5,0.5,0.1,0.5,1,1])
vis.display(INITIAL_CONFIG)
