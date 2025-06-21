import numpy as np
import timeit

def eval_dot():
    number_runs = 1000
    np_setup = ("import numpy as np;\n" 
                "a=np.random.randn(500,500);\n"
                "b=np.random.randn(500,500)")
    np_stmt = "np.dot(a,b)"
    np_time = timeit.timeit(np_stmt, np_setup, number=number_runs) / number_runs
    print (np_time)
    
def eval_matmul():
    pass 

if __name__ == "__main__":
    print("hi")
    eval_dot()