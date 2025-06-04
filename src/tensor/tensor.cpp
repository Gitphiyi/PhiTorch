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
            stride[i] = stride[i+1] * shape[i+1];
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
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->stride);
    free(t);
}

void print_metadata(Tensor* t) {
    printf("%d-dim %s Tensor: size = %d \n", t->ndim, t->device, t->dSize);
    printf("Stride = [");
    for(int i = 0; i < t->ndim; i++) {
        printf("%d", t->stride[i]);
        if(i < t->ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("Shape = [");
    for(int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if(i < t->ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

Tensor* flatten(const Tensor* t) {
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

Tensor* reshape(const Tensor* t, int* shape, int ndim) {
    int numData = 1;
    for(int i = 0; i < ndim; i++) {
        numData = numData * shape[i];
    }
    if(numData != t->dSize) {
        return nullptr;
    }
    Tensor* tens = (Tensor*) malloc(sizeof(Tensor));
    *tens = Tensor {
        .data = t->data,
        .grad = t->grad, 
        .shape = shape, 
        .stride = t->stride, 
        .ndim = ndim, 
        .dSize = t->dSize, 
        .device = t->device
    };
    return tens;
}

Tensor* transpose(const Tensor* t) {
    if(t->ndim != 2) {
        printf("Not a 2-D Tensor");
        return nullptr;
    }
    Tensor* tens = (Tensor*) malloc(sizeof(Tensor));
    tens->data = (float*) malloc(t->dSize * sizeof(float));
\   tens->grad = (float*) malloc(t->dSize * sizeof(float));
    tens->shape = (float*) malloc(t->ndim * sizeof(float));
    memcpy(tens->shape, t->shape, ndim * sizeof(float));
    tens->stride = (float*) malloc(t->ndim * sizeof(float));
    memcpy(tens->stride, t->stride, ndim * sizeof(float));
    tens->ndim = t->ndim;
    tens->dSize = t->dSize;
    tens->device = t->dSize;
    for(int i = 0; i < t->shape[0]; i++) {
        for(int j = 0; j < t->shape[1]; j++) {
            int src = t->stride[0] * i +t->stride[1] * j; 
            int target = t->stride[0] * j +t->stride[1] * i; 
            tens->data[target] = t->data[src];
            tens->grad[target] = t->grad[src];
            tens->data[src] = t->data[target];
            tens->grad[src] = t->grad[target];
        }
    }
    return tens;
    //multithread this if bigger than cache line
}

float dot(const Tensor* a, const Tensor* b) {
    float dot = 0;
    if(a->dSize != b->dSize || a->ndim != 1 || b->ndim != 1) {
        return NULL;
    }
    for(int i = 0; i < a->dSize; i++) {
        dot += a->data[i] * b->data[i];
    }
    return dot;
}

Tensor* matmul(const Tensor* a, const Tensor* b) {
    int* shape[2] = {a->shape[0], b->shape[1]};
    Tensor mat = create_tensor(shape, 2);
    if(a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0] ) {
        return NULL;
    }
    //foolish matrix multiplication
    for(int r = 0; r < a->shape[0]; r++) {
        for(int c = 0; c < a->shape[1]; c++) {

        }
    }
    return dot;
}

float item(const Tensor* t, int i) {
    if(i < 0 || i >= t->dSize) {
        return NULL;
    }
    return t->data[i];
}

float at(const int* shape, const int ndim) {

}

Tensor* equal(const Tensor* a, const Tensor* b) {
    
}