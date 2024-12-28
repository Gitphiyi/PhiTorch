import math
import numpy as np

version = "1.0.0"
author = "Philip Yi"

class Batch_Norm:
    def __init__(self, beta, gamma):
        pass 
    
class Activation:
    def __init__(self, activation_func: str):
        self.activation_func = 0 #use map to map str -> function
        pass
    def __call__(self, input: np.ndarray) -> np.ndarray:
        pass

class Linear:
    """
        Applies linear transformation 
        
        in_features: size of input sample
        out_features: size of output sample
        bias: If set to False, the layer will not learn an additive bias. 
    """
    def __init__(self, in_features:int, out_features:int, bias=True):
        bound = 1/ math.sqrt(in_features)
        self.weight = np.random.uniform(low = -bound, high=bound, size=(out_features, in_features))
        self.bias = np.random.uniform(low = -bound, high=bound, size=out_features)
                
    def __call__(self, input: np.ndarray) -> np.ndarray:
        weight_transposed = np.transpose(self.weight)
        if weight_transposed.ndim == 1:
            x = np.dot(input, weight_transposed)
        else: 
            x = input @ weight_transposed #matrix multiplication
        return x + self.bias #applies linear transformation
        
    def __str__(self):
        return f'Weight: {self.weight} \n Bias: {self.bias}'
