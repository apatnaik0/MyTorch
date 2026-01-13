import numpy as np
import scipy

class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function.
    """
    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = 1 / (1 + np.exp(-Z))  # TODO
        return self.A
    
    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = self.A * (1 - self.A)  # TODO
        dLdZ = dLdA * dAdZ  # TODO
        return dLdZ


class Tanh:
    """
    Tanh activation function.
    """
    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))  # TODO
        return self.A 
    
    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = 1 - self.A ** 2  # TODO
        dLdZ = dLdA * dAdZ  # TODO
        return dLdZ  


class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.
    """
    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = np.maximum(0, Z)  # TODO
        return self.A 
    
    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.where(self.A > 0, 1, 0)  # TODO
        dLdZ = dLdA * dAdZ  # TODO
        return dLdZ


class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    """
    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.Z = Z
        self.A = 0.5 * Z * (1 + scipy.special.erf(Z / np.sqrt(2)))  # TODO
        return self.A
    
    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = 0.5 * (1 + scipy.special.erf(self.Z / np.sqrt(2))) + (self.Z * np.exp(-0.5 * (self.Z ** 2)) / np.sqrt(2 * np.pi))  # TODO
        dLdZ = dLdA * dAdZ  # TODO
        return dLdZ 

class Swish:
    """
    Swish activation function.
    """
    def forward(self, Z, beta=1):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.Z = Z
        self.A = Z / (1 + np.exp(-beta * Z))  # TODO
        return self.A  
    
    def backward(self, dLdA, beta=1):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        sigmoid = 1 / (1 + np.exp(-beta * self.Z))
        dAdZ = sigmoid + beta * self.Z * sigmoid * (1 - sigmoid)  # TODO
        dLdZ = dLdA * dAdZ  # TODO
        self.dLdbeta = np.sum(dLdA * self.Z * self.Z * sigmoid * (1 - sigmoid)) # TODO
        return dLdZ 

class Softmax:
    """
    Softmax activation function.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        Note: How can we handle large overflow values? Hint: Check numerical stability.
        """
        self.A = np.exp(Z - np.max(Z, axis=1, keepdims=True)) / np.sum(np.exp(Z - np.max(Z, axis=1, keepdims=True)), axis=1, keepdims=True)  # TODO
        return self.A

    def backward(self, dLdA):
        # Calculate the batch size and number of features
        N = dLdA.shape[0]  # TODO
        C = dLdA.shape[1]  # TODO

        # Initialize the final output dLdZ with all zeros.
        dLdZ = np.zeros((N, C))  # TODO

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))  # TODO

            # Fill the Jacobian matrix.
            for m in range(C):
                for n in range(C):
                    J[m, n] = self.A[i, m] * (1 - self.A[i, m]) if m == n else -self.A[i, m] * self.A[i, n]  # TODO

            # Calculate the derivative of the loss with respect to the i-th input.
            dLdZ[i, :] = np.dot(dLdA[i, :], J)  # TODO

        return dLdZ   