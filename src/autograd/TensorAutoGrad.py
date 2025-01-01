import numpy as np

class Tensor:
    def __init__(self, data:np.array, children=[], op='', requires_grad = False, label = ''):
        self.label = label
        self.data = data 
        self.grad = 0
        self.children = set(children)
        self.op = op
        self._backward = lambda:None
        self.requires_grad = requires_grad
    
    def __repr__(self):
        labels = [x.label for x in self.children]
        return f"Tensor( data={self.data}, grad={self.grad}, children={labels}, op={self.op}, requires_grad={self.requires_grad} )"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert len(self.data) == len(other.data), "Tensor Lengths are not the same"
        r_grad = self.requires_grad or other.requires_grad
        data = self.data + other.data 
        output = Tensor(data, {self, other}, '+', r_grad)
        def backward():
            self.grad += output.grad
            other.grad += output.grad 
        output._backward = backward 
        return output 
    
    def __sub__(self, other):
        assert len(self.data) == len(other.data), "Tensor Lengths are not the same"
        other = other if isinstance(other, Tensor) else Tensor(other)
        r_grad = self.requires_grad or other.requires_grad
        data = self.data - other.data 
        output = Tensor(data, {self, other}, '-', r_grad)
        def backward():
            self.grad += output.grad
            other.grad -= output.grad 
        output._backward = backward 
        return output 
    
    def __mul__(self, other):
        assert len(self.data) == len(other.data), "Tensor Lengths are not the same"
        other = other if isinstance(other, Tensor) else Tensor(other)
        r_grad = self.requires_grad or other.requires_grad
        data = self.data * other.data 
        output = Tensor(data, {self, other}, '*', r_grad)
        def backward():
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data
        output._backward = backward 
        return output
    
    def __pow__(self, other):
        assert len(self.data) == len(other.data), "Tensor Lengths are not the same"

    
    def relu(self):
        """
            Potentially change this function so it does relu on top of linear transformation and calculate partial equation manually 
            without chunking the operation into multiple parts of the computation graph
            Worth calculating the time difference of how long it takes seperating and combining together
        """ 
        n = len(self.data)
        d = np.empty(n)
        for i in range(n): 
            d[i]=self.data[i] if self.data[i] >= 0 else 0
            
        output = Tensor(d,[self],'relu',self.requires_grad, f'relu_{self.label}')
        def backward():
            self.grad += (output.data > 0) * output.grad #avoids branch prediction
        output._backward = backward
        return output
    
    def MSELoss(self, target) -> float:    
        pass
    
    
    def backward(self):
        top_ord = []
        seen = set()
        def get_top(node : Tensor):
            if node in seen: 
                return 
            seen.add(node) 
            for c in node.children:
                get_top(c)
            top_ord.append(node)
        get_top(self)
        
        self.grad = np.ones_like(self.data)
        for node in reversed(top_ord):
            node._backward()
    
if __name__ == "__main__":
    a = np.array([1,2,3,4])
    b = np.array([2,2,2,2])
    Ta, Tb = Tensor(a, label='a'), Tensor(b, label='b')
    
    c = Ta-Tb 
    print(c)
    