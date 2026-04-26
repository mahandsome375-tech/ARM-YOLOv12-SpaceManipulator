








import math
import numpy as np


class TwoRFreeEnv:
    """
    State s = [x, y, xdot, ydot, ex, ey, q0, q1, q2, q0dot, q1dot, q2dot]
    Action a = [tau0, tau1, tau2]
    """

    def __init__(self, dt=0.01, ep_len=1000):
        self.dt = dt
        self.ep_len = ep_len


        self.ak1 = 0.8
        self.ak2 = 1.3
        self.ak3 = 1.5


        self.ad1 = 3.0
        self.ad2 = 1.0
        self.ad3 = 1.0
        self.ad4 = 20.0
        self.ad5 = 4.0
        self.ad6 = 1.0


        self.tau_max = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        self.dq_max = np.array([3.0, 3.0, 3.0], dtype=np.float32)




        self.A = 0.05
        self.x0 = 0.2
        self.y0 = 0.4
        self.w = 3.0


        self.t = 0.0
        self.step_count = 0
        self.q = np.zeros(3, dtype=np.float32)
        self.qdot = np.zeros(3, dtype=np.float32)
        self.s = np.zeros(12, dtype=np.float32)

        self._reset_state()


    def _fk(self, q):
        """
        Forward kinematics: given (q0,q1,q2), return the end-effector position (x,y)
        Use a standard three-link planar manipulator model
        """
        q0, q1, q2 = q
        ak1, ak2, ak3 = self.ak1, self.ak2, self.ak3

        x = (ak1 * math.cos(q0) +
             ak2 * math.cos(q0 + q1) +
             ak3 * math.cos(q0 + q1 + q2))
        y = (ak1 * math.sin(q0) +
             ak2 * math.sin(q0 + q1) +
             ak3 * math.sin(q0 + q1 + q2))
        return np.array([x, y], dtype=np.float32)

    def _jacobian(self, q):
        """
        End-effector Jacobian J(q) in R^{2x3}
        """
        q0, q1, q2 = q
        ak1, ak2, ak3 = self.ak1, self.ak2, self.ak3

        s0 = math.sin(q0)
        c0 = math.cos(q0)
        s01 = math.sin(q0 + q1)
        c01 = math.cos(q0 + q1)
        s012 = math.sin(q0 + q1 + q2)
        c012 = math.cos(q0 + q1 + q2)


        J11 = -ak1 * s0 - ak2 * s01 - ak3 * s012
        J12 = -ak2 * s01 - ak3 * s012
        J13 = -ak3 * s012


        J21 = ak1 * c0 + ak2 * c01 + ak3 * c012
        J22 = ak2 * c01 + ak3 * c012
        J23 = ak3 * c012

        J = np.array([[J11, J12, J13],
                      [J21, J22, J23]], dtype=np.float32)
        return J


    def _MC(self, q, qdot):
        """
        Construct M(q) and C(q,qdot) from Mbb/Mbm/Mmm and Cbb/Cbm/Cmb/Cmm in the MATLAB code
        q = [q0,q1,q2], qdot=[q0dot,q1dot,q2dot]
        """
        q0, q1, q2 = q
        q0dot, q1dot, q2dot = qdot
        a1, a2, a3 = self.ad1, self.ad2, self.ad3
        a4, a5, a6 = self.ad4, self.ad5, self.ad6

        c1 = math.cos(q1)
        c2 = math.cos(q2)
        c12 = math.cos(q1 + q2)
        s1 = math.sin(q1)
        s2 = math.sin(q2)
        s12 = math.sin(q1 + q2)


        Mbb = 2 * a1 * c1 + 2 * a2 * c2 + 2 * a3 * c12 + a4
        Mbm = np.array([
            a1 * c1 + 2 * a2 * c2 + a3 * c12 + a5,
            a2 * c2 + a3 * c12 + a6
        ], dtype=np.float32)
        Mmm = np.array([
            [2 * a2 * c2 + a5, a2 * c2 + a6],
            [a2 * c2 + a6,       a6]
        ], dtype=np.float32)

        M = np.zeros((3, 3), dtype=np.float32)
        M[0, 0] = Mbb
        M[0, 1:] = Mbm
        M[1:, 0] = Mbm
        M[1:, 1:] = Mmm


        Cbb = -a1 * s1 * q1dot - a2 * s2 * q2dot - a3 * s12 * (q1dot + q2dot)
        Cbm = np.array([
            -a1 * s1 * (q0dot + q1dot) - a2 * s2 * q2dot - a3 * s12 * (q0dot + q1dot + q2dot),
            -(a2 * s2 + a3 * s12) * (q0dot + q1dot + q2dot)
        ], dtype=np.float32)
        Cmb = np.array([
            a1 * s1 * q0dot - a2 * s2 * q2dot + a3 * s12 * q0dot,
            a2 * s2 * (q0dot + q1dot) + a3 * s12 * q0dot
        ], dtype=np.float32)
        Cmm = np.array([
            [-a2 * s2 * q2dot,         -a2 * s2 * (q0dot + q1dot + q2dot)],
            [ a2 * s2 * (q0dot + q1dot), 0.0]
        ], dtype=np.float32)

        C = np.zeros((3, 3), dtype=np.float32)
        C[0, 0] = Cbb
        C[0, 1:] = Cbm
        C[1:, 0] = Cmb
        C[1:, 1:] = Cmm

        return M, C


    def _ref(self, t):
        """
        Elliptic reference trajectory plus its first derivative as the reference velocity
        """
        xd = self.x0 + self.A * math.sin(self.w * t)
        yd = self.y0 + self.A * math.cos(self.w * t)

        xdot = self.A * self.w * math.cos(self.w * t)
        ydot = -self.A * self.w * math.sin(self.w * t)

        x_ref = np.array([xd, yd], dtype=np.float32)
        v_ref = np.array([xdot, ydot], dtype=np.float32)
        return x_ref, v_ref


    def _reset_state(self):

        self.q = np.array([0.0, math.pi / 6.0, math.pi / 3.0], dtype=np.float32)
        self.qdot = np.zeros(3, dtype=np.float32)
        self.t = 0.0
        self.step_count = 0


        x_ref, v_ref = self._ref(self.t)
        x = self._fk(self.q)
        J = self._jacobian(self.q)
        v = J @ self.qdot

        ex = x - x_ref
        exd = v - v_ref

        self.s = np.concatenate([x, v, ex, self.q, self.qdot], dtype=np.float32)

    def reset(self):
        self._reset_state()
        return self._obs()

    def _obs(self):
        return self.s.copy()

    def step(self, a):
        """
        a: np.ndarray shape (3,)
        Return: s', r, done, info
        """
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, -self.tau_max, self.tau_max)

        q = self.q.copy()
        qdot = self.qdot.copy()


        x_ref, v_ref = self._ref(self.t)
        x = self._fk(q)
        J = self._jacobian(q)
        v = J @ qdot

        ex = x - x_ref
        exd = v - v_ref


        M, C = self._MC(q, qdot)
        qdd = np.linalg.solve(M, a - C @ qdot)

        qdot = qdot + self.dt * qdd
        qdot = np.clip(qdot, -self.dq_max, self.dq_max)
        q = q + self.dt * qdot


        self.t += self.dt
        self.step_count += 1


        x_ref2, v_ref2 = self._ref(self.t)
        x2 = self._fk(q)
        J2 = self._jacobian(q)
        v2 = J2 @ qdot

        ex2 = x2 - x_ref2
        exd2 = v2 - v_ref2


        self.q = q.astype(np.float32)
        self.qdot = qdot.astype(np.float32)
        self.s = np.concatenate([x2, v2, ex2, self.q, self.qdot], dtype=np.float32)



        e_pos = np.linalg.norm(ex2)

        e_vel = np.linalg.norm(exd2)

        u_cost = 0.001 * float(a @ a)
        dq_cost = 0.001 * float(qdot @ qdot)

        reward = -(e_pos**2 + 0.1 * e_vel**2 + u_cost + dq_cost)

        done = (self.step_count >= self.ep_len)

        info = {
            "x": x2,
            "x_ref": x_ref2,
            "ex": ex2,
        }

        return self._obs(), float(reward), done, info
