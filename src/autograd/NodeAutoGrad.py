import math

class Node:
    def __init__(self, val, children=(), op=''):
        self.val = val 
        self.grad = 0
        self._backward = lambda:None
        self.children = set(children) #guarantees the children is a set. Children are just nodes that operations are done on to get val
        self.op = op
    
    def __repr__(self):
        return f"Node(val = {self.val}) \t # Children: {len(self.children)} \t Grad: {self.grad}"
    
    def __add__(self, other):
        output = Node(self.val+other.val, (self, other), '+')
        def backward():
            self.grad += output.grad
            other.grad += output.grad # it is not 1 due to chain rule
            #suppose L is loss, output = a, other = c. dL/dc = dL/da*da/dc. dL/da = output.grad and da/dc = 1. Thus dL/dc = output.grad           
        output._backward = backward
        return output
    
    def __mul__(self, other):
        output = Node(self.val * other.val, (self, other), '*')
        def backward():
            self.grad += output.grad * other.val
            other.grad += output.grad * self.val    
        output._backward = backward
        return output
    
    def __truediv__(self, other):
        assert other.val != 0, "Don't divide by 0"
        output = Node(self.val / other.val, (self, other), f'/{other}')
        def backward():
            self.grad += output.grad / other.val
            other.grad += output.grad * self.val / (other.val * other.val)  
        output._backward = backward
        return output
    
    def __pow__(self, other):
        output = Node(self.val ** other.val, (self, other), f'^{other}')
        def backward():
            self.grad += output.grad * other.val * self.val ** (other.val-1)
            other.grad += output.grad * math.log(self.val) * self.val ** other.val
        output._backward = backward
        return output
    
    def relu(self):
        output = Node(0 if self.val <= 0 else self.val, (self,), f'relu')
        def backward():
            self.grad += 0 if self.val <= 0 else output.grad
        output._backward = backward
        return output
        
    def backward(self):
        seen = set() 
        top_ordering = []
        def top_ord(v: Node):
            if v in seen:
                return
            seen.add(v)
            for c in v.children:
                top_ord(c) 
            top_ordering.append(v)
        top_ord(self) #this is list from nodes bottom up
        self.grad = 1
        for node in reversed(top_ordering):
            print(node)
            node._backward()
         
a0 = Node(1)
a1 = Node(2)
b0 = a0+a1
b0.relu()
b1 = a1*a0
c = b0*b1 
c.backward()
print(b0.grad)