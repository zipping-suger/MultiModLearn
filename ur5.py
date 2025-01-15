# reference: https://github.com/mc-capolei/python-Universal-robot-kinematics

import numpy as np
from numpy import linalg
import cmath
from math import cos, sin, atan2, acos, asin, sqrt, pi

class UR5RobotIK:
    def __init__(self):
        self.d1 = 0.1273
        self.a2 = -0.612
        self.a3 = -0.5723
        self.a7 = 0.075
        self.d4 = 0.163941
        self.d5 = 0.1157
        self.d6 = 0.0922

        self.d = np.matrix([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])
        self.a = np.matrix([0, -0.612, -0.5723, 0, 0, 0])
        self.alph = np.matrix([pi/2, 0, 0, pi/2, -pi/2, 0])

    def AH(self, n, th, c):
        T_a = np.matrix(np.identity(4), copy=False)
        T_a[0, 3] = self.a[0, n-1]
        T_d = np.matrix(np.identity(4), copy=False)
        T_d[2, 3] = self.d[0, n-1]

        Rzt = np.matrix([[cos(th[n-1, c]), -sin(th[n-1, c]), 0, 0],
                         [sin(th[n-1, c]), cos(th[n-1, c]), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], copy=False)

        Rxa = np.matrix([[1, 0, 0, 0],
                         [0, cos(self.alph[0, n-1]), -sin(self.alph[0, n-1]), 0],
                         [0, sin(self.alph[0, n-1]), cos(self.alph[0, n-1]), 0],
                         [0, 0, 0, 1]], copy=False)

        A_i = T_d * Rzt * T_a * Rxa
        return A_i

    def HTrans(self, th, c):
        A_1 = self.AH(1, th, c)
        A_2 = self.AH(2, th, c)
        A_3 = self.AH(3, th, c)
        A_4 = self.AH(4, th, c)
        A_5 = self.AH(5, th, c)
        A_6 = self.AH(6, th, c)

        T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6
        return T_06

    def invKine(self, desired_pos):
        th = np.matrix(np.zeros((6, 8)))
        P_05 = (desired_pos * np.matrix([0, 0, -self.d6, 1]).T - np.matrix([0, 0, 0, 1]).T)

        psi = atan2(P_05[1, 0], P_05[0, 0])
        # Clamp the value to the range [-1, 1] to avoid math domain errors
        phi = acos(np.clip(self.d4 / sqrt(P_05[1, 0]**2 + P_05[0, 0]**2), -1.0, 1.0))
        th[0, 0:4] = pi/2 + psi + phi
        th[0, 4:8] = pi/2 + psi - phi
        th = th.real

        cl = [0, 4]
        for i in range(len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = T_10 * desired_pos
            # Clamp the value to the range [-1, 1] to avoid math domain errors
            th[4, c:c+2] = +acos(np.clip((T_16[2, 3] - self.d4) / self.d6, -1.0, 1.0))
            th[4, c+2:c+4] = -acos(np.clip((T_16[2, 3] - self.d4) / self.d6, -1.0, 1.0))


        th = th.real

        cl = [0, 2, 4, 6]
        for i in range(len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = linalg.inv(T_10 * desired_pos)
            th[5, c:c+2] = atan2((-T_16[1, 2] / sin(th[4, c])), (T_16[0, 2] / sin(th[4, c])))

        th = th.real

        cl = [0, 2, 4, 6]
        for i in range(len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = self.AH(6, th, c)
            T_54 = self.AH(5, th, c)
            T_14 = (T_10 * desired_pos) * linalg.inv(T_54 * T_65)
            P_13 = T_14 * np.matrix([0, -self.d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T
            t3 = cmath.acos((linalg.norm(P_13)**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3))
            th[2, c] = t3.real
            th[2, c+1] = -t3.real

        cl = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = linalg.inv(self.AH(6, th, c))
            T_54 = linalg.inv(self.AH(5, th, c))
            T_14 = (T_10 * desired_pos) * T_65 * T_54
            P_13 = T_14 * np.matrix([0, -self.d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T

            th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(self.a3 * sin(th[2, c]) / linalg.norm(P_13))
            T_32 = linalg.inv(self.AH(3, th, c))
            T_21 = linalg.inv(self.AH(2, th, c))
            T_34 = T_32 * T_21 * T_14
            th[3, c] = atan2(T_34[1, 0], T_34[0, 0])

        th = th.real
        return th
    
    def sample_from_workspace(self, num_samples):
        samples = []
        for _ in range(num_samples):
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            z = np.random.uniform(0.0, 1.0)
            desired_pos = np.matrix([[1, 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]])
            samples.append(desired_pos)
        return samples

if __name__ == "__main__":
    ur5 = UR5RobotIK()
    desired_pos = np.matrix([[0, -1, 0, 0.5],
                             [1, 0, 0, 0.5],
                             [0, 0, 1, 0.5],
                             [0, 0, 0, 1]])

    joint_angles = ur5.invKine(desired_pos) # retuns a matrix of 6x8, the 8 columns are the 8 possible solutions
    print(joint_angles)