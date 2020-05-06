import numpy as np

class EKFonSimData:
    def __init__(self, Pxy, Pr, Qxy, Qr, Rd, numRob):
        self.Pxy = Pxy
        self.Pr = Pr
        self.Qxy = Qxy
        self.Qr = Qr
        self.Rd = Rd
        self.numRob = numRob
        self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))
        for i in range(self.numRob):
            for j in range(self.numRob):
                self.Pmatrix[0:2, 0:2, i, j] = np.eye(2)*Pxy
                self.Pmatrix[2, 2, i, j] = Pr

    def EKF(self, uNois, zNois, relativeState, ekfStride):
    # Calculate relative position between robot i and j in i's body-frame
        
        Q = np.diag([self.Qxy, self.Qxy, self.Qr, self.Qxy, self.Qxy, self.Qr])**2
        R = np.diag([self.Rd])**2  # observation covariance
        dtEKF = ekfStride*0.01
        for i in range(1):
            for j in [jj for jj in range(self.numRob) if jj!=i]:
                # the relative state Xij = Xj - Xi
                uVix, uViy, uRi = uNois[:, i]
                uVjx, uVjy, uRj = uNois[:, j]
                xij, yij, yawij = relativeState[:, i, j]
                dotXij = np.array([np.cos(yawij)*uVjx - np.sin(yawij)*uVjy - uVix + uRi*yij,
                                np.sin(yawij)*uVjx + np.cos(yawij)*uVjy - uViy - uRi*xij,
                                uRj - uRi])
                               
                statPred = relativeState[:, i, j] + dotXij * dtEKF #公式5（1）

                jacoF = np.array([[1,           uRi*dtEKF,  (-np.sin(yawij)*uVjx-np.cos(yawij)*uVjy)*dtEKF],
                                [-uRi*dtEKF,  1,          (np.cos(yawij)*uVjx-np.sin(yawij)*uVjy)*dtEKF ],
                                [0,           0,          1]]) #公式6 A
                jacoB = np.array([[-1,  0,  yij,  np.cos(yawij), -np.sin(yawij),  0],
                                [ 0, -1, -xij,  np.sin(yawij),  np.cos(yawij),  0],
                                [ 0,  0,   -1,                0,                0,  1]])*dtEKF #公式6 B
                
                PPred = jacoF@self.Pmatrix[:, :, i, j]@jacoF.T + jacoB@Q@jacoB.T #公式5 （2）
                xij, yij, yawij = statPred
                
                # X@H
                zPred = dist = np.sqrt(xij**2 + yij**2) #两机之间的距离 公式7  考虑高度！
                #zPred1=np.sqrt((xij-xik)**2 + (yij - yik**@))  
                #zPred2=bd

                jacoH = np.array([[xij/dist, yij/dist, 0]])
               # jacoH = np.array([[xij/dist, yij/dist, 0]
                                   # xik/dist1,yik/dist1]) # 公式8

                resErr = zNois[i, j] - zPred
                #resErr = zPred - zNois[i, j]
                #Err = zPred - jacoH@statPred.reshape((3,1))
                S = jacoH@PPred@jacoH.T + R
                K = PPred@jacoH.T@np.linalg.inv(S)#公式9 （1）              
                relativeState[:, [i], [j]] = statPred.reshape((3,1)) + K@np.array([[resErr]])#公式9 （2）
                self.Pmatrix[:, :, i, j] = (np.eye(len(statPred)) - K@jacoH)@PPred#公式9 （3）
        # print(np.trace(self.Pmatrix[:, :, i, j])) # for tuning the filter
        return relativeState

class EKFonRealData:
# This is slight different from the above one, cause the logging data repeat during some steps.
# Therefore, only different sensor data triggers a new update.
    def __init__(self, Pxy, Pr, Qxy, Qr, Rd, numRob):
        self.Pxy = Pxy
        self.Pr = Pr
        self.Qxy = Qxy
        self.Qr = Qr
        self.Rd = Rd
        self.numRob = numRob
        self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))
        for i in range(self.numRob):
            for j in range(self.numRob):
                self.Pmatrix[0:2, 0:2, i, j] = np.eye(2)*Pxy
                self.Pmatrix[2, 2, i, j] = Pr
        self.timeTick = np.zeros((self.numRob, self.numRob)) # to check difference
        self.zNoisOld = np.zeros((self.numRob, self.numRob)) # to check difference

    def EKF(self, uNois, zNois, relativeState):
    # Calculate relative position between robot i and j in i's body-frame
        Q = np.diag([self.Qxy, self.Qxy, self.Qr, self.Qxy, self.Qxy, self.Qr])**2
        R = np.diag([self.Rd])**2  # observation covariance
        uNois[0:2,:] = 0.88 * uNois[0:2,:] # a proportion gain, seems not important
        for i in range(1):
            for j in [jj for jj in range(self.numRob) if jj!=i]:
                if self.zNoisOld[i, j] != zNois[i, j]:
                    self.zNoisOld[i, j] = zNois[i, j]
                    dtEKF = 0.01*self.timeTick[i, j]
                    self.timeTick[i, j] = 1
                    # the relative state Xij = Xj - Xi
                    uVix, uViy, uRi = uNois[:, i]
                    uVjx, uVjy, uRj = uNois[:, j]
                    xij, yij, yawij = relativeState[:, i, j]
                    dotXij = np.array([np.cos(yawij)*uVjx - np.sin(yawij)*uVjy - uVix + uRi*yij,
                                    np.sin(yawij)*uVjx + np.cos(yawij)*uVjy - uViy - uRi*xij,
                                    uRj - uRi])
                    statPred = relativeState[:, i, j] + dotXij * dtEKF

                    jacoF = np.array([[1,       uRi*dtEKF,  (-np.sin(yawij)*uVjx-np.cos(yawij)*uVjy)*dtEKF],
                                    [-uRi*dtEKF, 1,         (np.cos(yawij)*uVjx-np.sin(yawij)*uVjy)*dtEKF ],
                                    [0,          0,         1]])
                    jacoB = np.array([[-1,  0,  yij,  np.cos(yawij), -np.sin(yawij),  0],
                                    [ 0, -1, -xij,  np.sin(yawij),  np.cos(yawij),  0],
                                    [ 0,  0,   -1,                0,                0,  1]])*dtEKF
                    PPred = jacoF@self.Pmatrix[:, :, i, j]@jacoF.T + jacoB@Q@jacoB.T
                    xij, yij, yawij = statPred
                    zPred = dist = np.sqrt(xij**2 + yij**2)+0.0001
                    jacoH = np.array([[xij/dist, yij/dist, 0]])
                    resErr = zNois[i, j] - zPred
                    S = jacoH@PPred@jacoH.T + R
                    K = PPred@jacoH.T@np.linalg.inv(S)
                    relativeState[:, [i], [j]] = statPred.reshape((3,1)) + K@np.array([[resErr]])
                    self.Pmatrix[:, :, i, j] = (np.eye(len(statPred)) - K@jacoH)@PPred
                else:
                    self.timeTick[i, j] += 1
        # print(np.trace(self.Pmatrix[:, :, i, j])) # for tuning the filter
        return relativeState