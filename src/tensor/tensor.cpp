// #pragma once
#include "Tensor.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// #include <string.h>

// using namespace std;
Tensor* create_tensor(int* shape, int ndim) {
    int dSize = 0;
    int* stride = (int*) malloc(ndim * sizeof(int));

    for(int i = ndim-1; i >= 0; i--) {
        dSize += shape[i];
        if( i < ndim-1) {
            stride[i] = stride[i+1] * shape[i];
        } 
        else {
            stride[i] = 1;
        }
    }
    float* data = (float*) malloc(dSize * sizeof(float));
    float* grad = (float*) malloc(dSize * sizeof(float));;
    Tensor* t = (Tensor*) malloc(sizeof(Tensor));
    *t = Tensor {
        .data = data,
        .grad = grad, 
        .shape = shape, 
        .stride = stride, 
        .ndim = ndim, 
        .dSize = dSize, 
        .device = "cpu"
    };
    return t;
}
void delete_tensor(Tensor* t) {
    free(t->data);
    free(t->grad);
    free(t->stride);
    free(t);
}

void print(Tensor* t) {
    printf("Tensor: \n");
    for(int i = 0; i < t->ndim; i++) {
        printf("i\n");
    }
}

Tensor* flatten(Tensor* t) {
    int shape[1] = { t->dSize };
    Tensor* tens = (Tensor*) malloc(sizeof(Tensor));
    *tens = Tensor {
        .data = t->data,
        .grad = t->grad, 
        .shape = shape, 
        .stride = t->stride, 
        .ndim = 1, 
        .dSize = t->dSize, 
        .device = t->device
    };
    return tens;
}