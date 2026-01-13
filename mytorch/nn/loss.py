import numpy as np


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)
        """
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]  
        self.C = self.A.shape[1]  
        se = (self.A-self.Y)*(self.A-self.Y)  
        sse = np.ones((1,self.N)) @ se @ np.ones((self.C,1))  
        mse = sse / (self.N * self.C)
        return mse 

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.
        """
        dLdA = 2*(self.A-self.Y) / (self.N * self.C)  
        return dLdA  


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)
        """
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]  
        self.C = self.A.shape[1]  

        Ones_C = np.ones((self.C, 1))  
        Ones_N = np.ones((self.N, 1))  
        self.softmax = np.exp(self.A) / np.sum(np.exp(self.A), axis=1, keepdims=True) 

        crossentropy = (- self.Y * np.log(self.softmax)) @ Ones_C  
        sum_crossentropy_loss = Ones_N.T @ crossentropy  
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        return mean_crossentropy_loss  

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.
        """
        dLdA = (self.softmax - self.Y)/self.N  
        return dLdA  