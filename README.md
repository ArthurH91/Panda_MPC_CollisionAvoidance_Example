# Panda_MPC_Collision_Avoidance_Example

This repo is a batch of examples of collision avoidance for trajectory optimisation and model predictive control (MPC).
It has several dependencies : 


# Dependencies 

## For OCP & MPC scripts : 

- HPPFCL : https://github.com/humanoid-path-planner/hpp-fcl/tree/hppfcl3x **(HPPFCL3X BRANCH REQUIERED)** for collision computations.
- Pinocchio: https://github.com/stack-of-tasks/pinocchio fast rigid body dynamics. **(if you have pinocchio3, switch to the pinocchio3 branch)**
- Crocoddyl: https://github.com/loco-3d/crocoddyl framework for the solver.
- MiM Solvers: https://github.com/machines-in-motion/mim_solvers solver.
- Mim Robots: https://github.com/machines-in-motion/mim_robots pybullet env.
- Colmpc: https://github.com/ArthurH91/colmpc collision residual for the solver.

## For visualization : 
- Meshcat: https://github.com/meshcat-dev/meshcat

# Installations

HPP-FCL & Pinocchio must be built from sources. Don't forget to checkout the hppfcl3x branch. Build pinocchio with the flag : WITH_COLLISION_SUPPORT=ON. 

Setting aside Colmpc which must be built from source, the other packages can be built either with conda or whatever you want to use for it. Mim Robot is built with pip.

# Usage


Before trying the scripts, test your hppfcl installation. To do this and make sure the hppfcl librairy works well in your computer, run : 
``` python tests/__init__.py```.

## For the trajectory optimisation part:

To try the examples, create a meshcat server using a terminal and the following command : ```meshcat-server```. In another terminal, you can launch for instance ```python traj-optimisation/demo_panda_reaching_single_obs_capsule_capsule.py``` to run the demo.

## For the MPC part:

Simply run ```python mpc/mpc_kuka_reaching_scene1.py```

As the code is still in developpement, the code is constantly moving and sometimes, examples do not work. Hence, do not hesitate to contact me at ahaffemaye@laas.fr. 

# Credits

This repo is based on https://github.com/machines-in-motion/minimal_examples_crocoddyl/tree/master from Sebastien Kleff. 

