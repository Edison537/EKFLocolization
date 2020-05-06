
import numpy as np

class newEKF :
    def __init__(self, Pxy, Pr, Qxy, Qr, Rd, numRob):
        self.Pxy = Pxy
        self.Pr = Pr
        self.Qxy = Qxy
        self.Qr = Qr
        self.Rd = Rd
        self.numRob = numRob
        self.Pmatrix = np.zeros((self.numRob+1, 2, 2))
        #self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))
        #for i in range(self.numRob):
            #for j in range(self.numRob):
                #self.Pmatrix[0:2, 0:2, i, j] = np.eye(2)*Pxy
                #self.Pmatrix[2, 2, i, j] = Pr

    def nEKF(self, uNois, zNois, Esti, ekfStride, droneID):

        #应该要改动
        Q = np.diag([self.Qxy, self.Qxy])**2
        R = np.diag([self.Rd, self.Rd, self.Rd, self.Rd])**2  # observation covariance
        dtEKF = ekfStride*0.01

        uVix, uViy, uRi = uNois[:, droneID]
        dotXij = np.array([uVix, uViy, uRi])

        statPred = Esti[:, droneID] + dotXij * dtEKF #公式5（1）

        jacoF=np.array([[1, dtEKF],
                        [0, 1]])

        #jacoB应该不用了
        PPred = jacoF@self.Pmatrix[droneID, :, :]@jacoF.T + Q#公式5 （2）

        xij, yij, yawij = statPred
 
        zPred = []
        jacoH = []
        for i in [jj for jj in range(self.numRob)]:      
            zPred.insert(i, np.sqrt((xij - Esti[0, i])**2 + (yij - Esti[1, i])**2)) 
           
        for i in [jj for jj in range(self.numRob)]:
            xi = (xij-Esti[0, i])/zPred[i]
            xj = (yij-Esti[1, i])/zPred[i]
            jacoH.insert(i, [xi, xj]) 
        
        
        jacoH = np.array(jacoH)
        resErr = zNois[droneID, :] - zPred
        S = jacoH@PPred@jacoH.T + R
        K = PPred@jacoH.T@np.linalg.inv(S)
        statPred = statPred.tolist()       
        K = np.array(K)
        Esti[0:2, droneID] = statPred[0:2]+ K@resErr#公式9 （2）
        self.Pmatrix[droneID, :, :] = (np.eye(len(statPred[0:2])) - K@jacoH)@PPred#公式9 （3）
        #print(self.Pmatrix[1, 0, :])
        
        return Esti





