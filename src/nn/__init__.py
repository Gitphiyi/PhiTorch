import numpy as np
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.autograd.TensorAutoGrad import Tensor 

class BatchNorm:
    """
        Batch Normalization will act as a layer transformation to help normalize a "mini-batch" which is equivalent to a layer
        
        size: 
        affine: allows BN transform to represent identity transform. Detailed in bottom of page 4 of the paper
    """
    def __init__(self, size:int, affine:bool = False):
        self.affine = affine
        self.gamma = Tensor(np.random.normal(size=(size, size)), requires_grad=True)
        self.beta = Tensor(np.zeros(size), requires_grad=True)
    
    def __call__(self, input:Tensor) -> Tensor:
        if self.affine:
            return input 
        mean = np.mean(input.data)
        var = np.var(input.data)
        input = (input - mean) / var # Likely will throw error since I haven't created scallar subtract and divide
        out = self.gamma @ input # will throw error since I also did not write what matmul is
        output = out + self.beta
        return output
