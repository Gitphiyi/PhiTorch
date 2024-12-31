import numpy as np
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.autograd.TensorAutoGrad import Tensor


def add_test():
    a = np.array([1,2,3,4])
    b = np.array([2,2,2,2])
    Ta, Tb = Tensor(a, label='a'), Tensor(b, label='b')
    c = Ta+Tb
    print("Add Test Successful") if np.array_equal(c.data, np.array([3,4,5,6])) else print("Failed Add Test")
    
def sub_test():
    a = np.array([1,2,3,4])
    b = np.array([2,2,2,2])
    Ta, Tb = Tensor(a, label='a'), Tensor(b, label='b')
    c = Ta-Tb
    print("Subtract Test Successful") if np.array_equal(c.data, np.array([-1,0,1,2])) else print("Failed Subtract Test")
    
def mul_test():
    a = np.array([1,2,3,4])
    b = np.array([2,2,2,2])
    Ta, Tb = Tensor(a, label='a'), Tensor(b, label='b')
    c = Ta*Tb
    print("Mult Test Successful") if np.array_equal(c.data, np.array([2,4,6,8])) else print("Mult Subtract Test")

def relu_test():
    a = np.array([0,1,-1,3])
    Ta = Tensor(a, label='a')
    c = Ta.relu()
    print(c)
    #assert np.array_equal(a.data, np.array([0,1,0,3])), "Incorrect data values"
    assert len(c.children) == 1, "Incorrect children length"
    
def backward_test():
    a0 = np.array([0,1,-1,3])
    a1 = np.array([1,1,-1,2])
    Ta0, Ta1 = Tensor(a0, label='a0'), Tensor(a1, label='a1')
    b0 = Ta0*Ta1
    b0.label = "b0"
    b1 = Ta0+Ta1 
    b1.label = "b1"
    c = b0*b1 
    c.label = "c"
    c.backward()
    
    print(c)
    print(b0)
    print(b1)
    print(Ta1)



if __name__ == "__main__":
    #add_test()
    #sub_test()
    #mul_test()
    #relu_test()
    backward_test()