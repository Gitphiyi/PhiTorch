import numpy as np
from ..autograd.TensorAutoGrad import Tensor 
from nn import BatchNorm

class Module:
    def zero_grad(self):
        for v in self.params():
            v.grad = np.zeros_like(v.data)
            
    def parameters(self):
        return []

class Layer(Module):  
    def __init__(self, nin, nout, act_func, batch_norm:bool = True):
        self.w = Tensor(np.random.normal(size=(nout, nin)), requires_grad=True) 
        self.bias = Tensor(np.zeros(nout), requires_grad=True) 
        self.act_func = act_func
        self.batch_norm = batch_norm

    def __call__(self, input: Tensor):
        out1 = self.w @ input
        out2 = out1 + self.bias
        b = BatchNorm(len(out2.data))
        temp = b(out2) if self.batch_norm else out2
        output = self.act_func(temp)
        return output
    
    def parameters(self):
        return [self.w, self.bias]
    
class MLP(Module):
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def __call__(self, input):
        """ 
            input: Tensor with no requires_grad off
        """
        output = input
        for layer in self.layers:
            output = layer()
        return output
    
    def parameters(self):
        out = []
        for _ in self.layers:
            out.extend(_.parameters())
        return out