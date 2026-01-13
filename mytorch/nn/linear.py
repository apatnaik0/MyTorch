import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros for `W` and `b`.
        """
        self.debug = debug
        self.W = np.zeros((out_features, in_features)) 
        self.b = np.zeros((out_features, 1)) 

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)
        """
        self.A = np.array(A)  
        self.N = self.A.shape[0]  

        # To help with broadcasting
        self.ones = np.ones((self.N, 1))

        Z = self.A @ self.W.T + self.ones @ self.b.T  
        return Z  

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)
        """
        dLdA = dLdZ @ self.W  
        self.dLdW = dLdZ.T @ self.A  
        self.dLdb = dLdZ.T @ self.ones  

        if self.debug:
            self.dLdA = dLdA

        return dLdA
