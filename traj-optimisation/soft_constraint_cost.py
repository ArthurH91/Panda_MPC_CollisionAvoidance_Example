from copy import copy
import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl.utils import *

import hppfcl


class CostModelPairCollision(crocoddyl.CostModelAbstract):
    def __init__(
        self,
        state,
        geom_model: pin.Model,
        geom_data,
        pair_id: int,
        joint_id: int,
    ):
        """_summary_

        Args:
            state (crocoddyl.StateMultibody): _description_
            geom_model (pin.CollisionModel): _description_
            pair_id (tuple): _description_
            joint_id (int): _description_
        """
        r_activation = 3

        crocoddyl.CostModelAbstract.__init__(
            self, state, crocoddyl.ActivationModelQuad(r_activation)
        )

        self.pinocchio = self.state.pinocchio
        self.geom_model = geom_model
        self.pair_id = pair_id
        self.joint_id = joint_id
        self.geom_data = geom_data

    def calc(self, data, x, u=None):
        self.q = np.array(x[: self.state.nq])
        # computes the distance for the collision pair pair_id
        pin.forwardKinematics(self.pinocchio, data.shared.pinocchio, self.q)
        pin.updateGeometryPlacements(
            self.pinocchio,
            data.shared.pinocchio,
            self.geom_model,
            self.geom_data,
            self.q,
        )

        self.req = hppfcl.DistanceRequest()
        self.res = hppfcl.DistanceResult()
        self.shape1_id = self.geom_model.collisionPairs[self.pair_id].first
        self.shape1_geom = self.geom_model.geometryObjects[self.shape1_id].geometry
        self.shape1_placement = self.geom_data.oMg[self.shape1_id]

        self.shape2_id = self.geom_model.collisionPairs[self.pair_id].second
        self.shape2_geom = self.geom_model.geometryObjects[self.shape2_id].geometry
        self.shape2_placement = self.geom_data.oMg[self.shape2_id]

        data.d = hppfcl.distance(
            self.shape1_geom,
            self.shape1_placement,
            self.shape2_geom,
            self.shape2_placement,
            self.req,
            self.res,
        )
        
        # calculate residual
        if self.res.min_distance <= 0:
            data.residual.r[:] = self.res.getNearestPoint1() - self.res.getNearestPoint2()
            data.cost = 0.5 * np.dot(data.residual.r[:], data.residual.r[:])

        else:
            data.residual.r[:].fill(0.0)
            data.cost= 0
        
        self.res_numdiff = data.residual.r[:]
            
    def calcDiff(self, data, x, u=None):
        q = x[:self.pinocchio.nq]

        if self.res.min_distance <= 0:

            pin.updateGeometryPlacements(
                self.pinocchio,
                data.shared.pinocchio,
                self.geom_model,
                self.geom_data,
                q,
            )
            
            jacobian1 = pin.computeFrameJacobian(
                self.pinocchio,
                data.shared.pinocchio,
                q,
                self.shape1.parentFrame,
                pin.LOCAL_WORLD_ALIGNED,
            )
            # print("python")
            jacobian2 = pin.computeFrameJacobian(
                self.pinocchio,
                data.shared.pinocchio,
                q,
                self.shape2.parentFrame,
                pin.LOCAL_WORLD_ALIGNED,
            )
            # print(f"parentFrame py : {self.shape2.parentFrame} ")
            # print(f"J2 py :{jacobian2}")

            cp1 = self.res.getNearestPoint1()
            cp2 = self.res.getNearestPoint2()
            
            # print("------------------")
            # print("python : ")
            # print(f"q : {q}")
            # print(f"cp 1 : {cp1}")
            # print(f"cp 2 : {cp2}")
            # print(f"distance : {distance}")
            ## Transport the jacobian of frame 1 into the jacobian associated to cp1
            # Vector from frame 1 center to p1
            f1p1 = cp1 - data.shared.pinocchio.oMf[self.shape1.parentFrame].translation
            # print(f"f1p1 py : {f1p1}")
            # The following 2 lines are the easiest way to understand the transformation
            # although not the most efficient way to compute it.
            f1Mp1 = pin.SE3(np.eye(3), f1p1)
            
            jacobian1 = f1Mp1.actionInverse @ jacobian1
            # print(f"J1 py :{jacobian1}")

            ## Transport the jacobian of frame 2 into the jacobian associated to cp2
            # Vector from frame 2 center to p2
            f2p2 = cp2 - data.shared.pinocchio.oMf[self.shape2.parentFrame].translation
            # The following 2 lines are the easiest way to understand the transformation
            # although not the most efficient way to compute it.
            f2Mp2 = pin.SE3(np.eye(3), f2p2)
            jacobian2 = f2Mp2.actionInverse @ jacobian2
            #     # compute the residual derivatives
            # data.residual.Rx[:3, :self.pinocchio.nq] = jacobian1[:3] - jacobian2[:3]
            data.residual.Rx[:3, :self.pinocchio.nq] = self.numdiff(self.calc, x, data)
        else:
            data.residual.Rx[:3, :self.pinocchio.nq].fill(0.0)
        
        
        data.Lx[:] = np.dot(data.residual.Rx.T, data.residual.r)
        data.Lxx[:] = np.dot(data.residual.Rx.T, data.residual.Rx)
        
    def numdiff(self, f, x, data, eps=1e-8):
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
        f(data, x)
        f0 = np.copy(self.res_numdiff)
        res = []
        for i in range(self.state.nq):
            xc[i] += eps
            f(data, xc)
            fc = self.res_numdiff
            res.append(copy(fc - f0) / eps)
            xc[i] = x[i]
        return np.array(res).T

           
    def createData(self, collector):
        data = CostDataPairCollision(self, collector)
        return data


class CostDataPairCollision(crocoddyl.CostDataAbstract):
    def __init__(self, model, data_collector):
        crocoddyl.CostDataAbstract.__init__(self, model, data_collector)
        self.d = np.zeros(3)


if __name__ == "__main__":
    pass
