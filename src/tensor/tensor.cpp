#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>

typedef struct {
    float* data; // array of data
    int* shape; // array of shape of each dimension
    int ndim; // number of dimensions (rank)
    std::string device; //cpu/gpu
} Tensor;


int main() {
    return 1;
}