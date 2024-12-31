import numpy as np

class Tensor:
    def __init__(self):
        self.data = []
        self.grad = []
        self.grad_fn = None
        self.requires_grad = False
        self.is_leaf = False
        