import numpy as np


class BatchNorm1d:
    """
    Batch Normalization layer for 1D inputs.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.
        """
        self.Z = Z
        self.N = self.Z.shape[0]  # Calculate batch size
        self.M = np.mean(self.Z, axis=0)  # Calculate mini-batch mean
        self.V = np.var(self.Z, axis=0)  # Calculate mini-batch variance

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)  # Calculate the normalized input Ẑ
            self.BZ = self.BW * self.NZ + self.Bb  # Calculate the scaled and shifted for the normalized input Ẑ

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M  # Calculate running mean
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V  # Calculate running variance
        else:
            # inference mode
            self.NZ = (self.Z - self.running_M) / np.sqrt(self.running_V + self.eps)  # Calculate the normalized input Ẑ using the running average for mean and variance
            self.BZ = self.BW * self.NZ + self.Bb  # Calculate the scaled and shifted for the normalized input Ẑ

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).
        """
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)  # Sum over the batch dimension.
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)  # Scale gradient of loss wrt BatchNorm transformation by normalized input NZ.

        dLdNZ = dLdBZ * self.BW  # Scale gradient of loss wrt BatchNorm transformation output by gamma (scaling parameter).

        dLdV = - 0.5 * np.sum(dLdNZ * (self.Z - self.M) * np.power(self.V + self.eps, -1.5), axis=0, keepdims=True)  # Compute gradient of loss backprop through variance calculation.
        dNZdM = -np.power(self.V + self.eps, -0.5) - 0.5 * (self.Z - self.M) * np.power(self.V + self.eps, -1.5) * (-2/self.N * np.sum(self.Z - self.M, axis=0, keepdims=True)) # Compute derivative of normalized input with respect to mean.
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)  # Compute gradient of loss with respect to mean.

        dLdZ = dLdNZ * np.power(self.V + self.eps, -0.5) + dLdM / self.N + dLdV * 2 * (self.Z - self.M) / self.N  #  Compute gradient of loss with respect to the input.
        return dLdZ 
