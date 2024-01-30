import pinocchio
from sys import argv
from os.path import dirname, join, abspath
 
from wrapper_meshcat import MeshcatWrapper


pinocchio_model_dir1 = join(
    dirname(dirname(dirname(str(abspath(__file__))))), "models"
)
mesh_dir = pinocchio_model_dir1 + "/franka_description/meshes/"
print(mesh_dir)
# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = dirname(str(abspath(__file__)))
 
# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = pinocchio_model_dir + '/robot(1).urdf' if len(argv)<2 else argv[1]
print(urdf_filename)
# Load the urdf model
model, cmodel, vmodel    = pinocchio.buildModelsFromUrdf(urdf_filename, [], pinocchio.JointModelFreeFlyer())
print('model name: ' + model.name)
 
# Create data required by the algorithms
data     = model.createData()
 
# Sample a random configuration
q        = pinocchio.randomConfiguration(model)
print('q: %s' % q.T)
 
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model,data,q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format( name, *oMi.translation.T.flat )))


# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis, meshcatVis = MeshcatVis.visualize(
    robot_model=model,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
)
# Displaying the initial
vis.display(q)

input()