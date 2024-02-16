from os.path import dirname, join, abspath
import numpy as np

import pinocchio as pin
import hppfcl
import copy
from wrapper_meshcat import MeshcatWrapper
from wrapper_robot import RobotWrapper

from utils import BLUE, YELLOW_FULL

## HELPERS 

### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 1.55]))
### CREATING THE OBSTACLE
OBSTACLE_RADIUS = 1.5e-1
OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([0.25, -0.425, 1.5])

def panda_loader():
    ### LOADING THE ROBOT
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)
    srdf_model_path = model_path + "/panda/demo.srdf"

    # Creating the robot
    robot_wrapper = RobotWrapper(
        urdf_model_path=urdf_model_path, mesh_dir=mesh_dir, srdf_model_path=srdf_model_path
    )
    rmodel, cmodel, vmodel = robot_wrapper()
    
    OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE,
        OBSTACLE_POSE,
    )
    OBSTACLE_GEOM_OBJECT.meshColor = BLUE
    IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)
    ## ADDING THE COLLISION PAIR BETWEEN A LINK OF THE ROBOT & THE OBSTACLE
    cmodel.geometryObjects[cmodel.getGeometryId("panda2_link7_sc_2")].meshColor = YELLOW_FULL
    cmodel.addCollisionPair(
        pin.CollisionPair(cmodel.getGeometryId("panda2_link7_sc_2"), IG_OBSTACLE)
    )
    return rmodel, cmodel, vmodel

def compute_1p_dot_1(q):
    pin.computeAllTerms(rmodel, rdata, q, vq)
    cp1_se3 = pin.SE3.Identity()
    cp1_se3.translation = cp1
    p1 = (rdata.oMi[joint1_idx].inverse() * cp1_se3).translation
    v1 = rdata.v[joint1_idx].linear
    rw1 = np.cross(p1,rdata.v[joint1_idx].angular)
    return v1 + rw1

def compute_1p_dot_1_deriv(q):
    pin.computeAllTerms(rmodel, rdata, q, vq)
    cp1_se3 = pin.SE3.Identity()
    cp1_se3.translation = cp1
    p1 = (rdata.oMi[joint1_idx].inverse() * cp1_se3).translation
    jvd = pin.getJointVelocityDerivatives(rmodel, rdata, joint1_idx, pin.WORLD)
    v_dot = jvd[0][:3]
    w_dot = jvd[0][3:]
    return v_dot + np.matmul(p1, w_dot)

def numdiff(f, x, eps=1e-8):
    """Estimate df/dx at x with finite diff of step eps

    Parameters
    ----------
    f : function handle
        Function evaluated for the finite differente of its gradient.
    x : np.ndarray
        Array at which the finite difference is calculated
    eps : float, optional
        Finite difference step, by default 1e-6

    Returns
    -------
    jacobian : np.ndarray
        Finite difference of the function f at x.
    """
    xc = np.copy(x)
    f0 = np.copy(f(x))
    res = []
    for i in range(len(x)):
        xc[i] += eps
        res.append(copy.copy(f(xc) - f0) / eps)
        xc[i] = x[i]
    return np.array(res).T

if __name__ =="__main__":
    
    rmodel, cmodel, vmodel = panda_loader()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
        # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis, meshcatVis = MeshcatVis.visualize(
        TARGET_POSE,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
)
    # Config & velocity
    q = np.array([ 0.439,   0.9274 , 0.3113 , 0.3734 ,-0.2116,  1.1214 , 0.024])
    vq = np.array([1,1,1,1,1,1,1])
    
    vis.display(q)
    
    pin.computeAllTerms(rmodel, rdata, q, vq)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    
    joint1_idx = cmodel.geometryObjects[cmodel.getGeometryId("panda2_link7_sc_2")].parentJoint

    geom_id1_idx = cmodel.getGeometryId("panda2_link7_sc_2")
    geom_id2_idx = cmodel.getGeometryId("obstacle")
    # Hppfcl distance
    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    
    hppfcl.distance(
        cmodel.geometryObjects[geom_id1_idx].geometry,
        cdata.oMg[geom_id1_idx],
        cmodel.geometryObjects[geom_id2_idx].geometry,
        cdata.oMg[geom_id2_idx],
        req, 
        res
    )

    cp1 = res.getNearestPoint1()

    print(f"1p_dot_1: {compute_1p_dot_1(q)}")
    print(f"numdiff: {numdiff(compute_1p_dot_1, q)}")
    print(f"deriv 1p_dot_1_: {compute_1p_dot_1_deriv(q)}")
    print(f"diff: {numdiff(compute_1p_dot_1, q) - compute_1p_dot_1_deriv(q)}")