#include "Tensor.h"
#include "../lib/ThreadPool.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
// #include <string.h>


// using namespace std;
extern thread_pool_t* threads;

Tensor* create_tensor(int* shape, int ndim) {
    int dSize = 1;
    int* stride = (int*) malloc(ndim * sizeof(int));

    for(int i = ndim-1; i >= 0; i--) {
        dSize *= shape[i];
        if( i < ndim-1) {
            stride[i] = stride[i+1] * shape[i+1];
        } 
        else {
            stride[i] = 1;
        }
    }
    float* data = (float*) malloc(dSize * sizeof(float));
    float* grad = (float*) malloc(dSize * sizeof(float));;
    Tensor* tens = (Tensor*) malloc(sizeof(Tensor));
    tens->data = data;
    tens->grad = grad;
    tens->shape = shape;
    tens->stride = stride;
    tens->ndim = ndim;
    tens->dSize = dSize;
    tens->device = "cpu";
    return tens;
}

void delete_tensor(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->stride);
    free(t);
}

int set_data(Tensor* tens, float* data, int size) {
    if( size != tens->dSize) {
        printf("dSize: %d != size: %d\n", tens->dSize, size);
        return -1;
    }
    for(int i = 0; i < size; i ++) {
        tens->data[i] = data[i];
    }
    return 0;
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
    tens->data = t->data;
    tens->grad = t->grad;
    tens->shape = shape;
    tens->stride = t->stride;
    tens->ndim = 1;
    tens->dSize = t->dSize;
    tens->device = t->device;
    return tens;
}

Tensor* reshape(const Tensor* t, int* shape, int ndim) {
    int numData = 1;
    for(int i = 0; i < ndim; i++) {
        numData = numData * shape[i];
    }
    if(numData != t->dSize) {
        return NULL;
    }
    Tensor* tens = (Tensor*) malloc(sizeof(Tensor));
    tens->data = t->data;
    tens->grad = t->grad;
    tens->shape = shape;
    tens->stride = t->stride;
    tens->ndim = ndim;
    tens->dSize = t->dSize;
    tens->device = t->device;
    return tens;
}

Tensor* transpose(const Tensor* t) {
    if(t->ndim != 2) {
        printf("Not a 2-D Tensor");
        return NULL;
    }
    Tensor* tens = (Tensor*) malloc(sizeof(Tensor));
    tens->data = (float*) malloc(t->dSize * sizeof(float));
    tens->grad = (float*) malloc(t->dSize * sizeof(float));
    tens->shape = (int*) malloc(t->ndim * sizeof(float));
    memcpy(tens->shape, t->shape, t->ndim * sizeof(float));
    tens->stride = (int*) malloc(t->ndim * sizeof(float));
    memcpy(tens->stride, t->stride, t->ndim * sizeof(float));
    tens->ndim = t->ndim;
    tens->dSize = t->dSize;
    tens->device = t->device;
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

/**
 * Dot product process can be split into two steps: 
 * Multiplying pairwise a and b values
 * Summing the products into one value
 * 
 */
typedef struct dot_t {
    const Tensor* a;
    const Tensor* b;
    const int st;
    const int end;
} dot_t;
void* dot_helper(void* args) {
    dot_t* d = (dot_t*) args;
    const int maxLoopIter = 5;
    const int mid = (d->end + d->st) / 2;
    float* result = (float*) malloc(sizeof(float));
    *result = 0.0;
    //Base Case
    if (d->end - d->st <= maxLoopIter) {
        for(int i = d->st; i <= d->end; i++) {
            printf("num %d: %f\n", i, d->a->data[i]);
            *result += d->a->data[i] * d->b->data[i];
        }
        return result;
    }

    //Thread step
    pthread_t thread1;
    pthread_t thread2;
    dot_t* l_tens = malloc(sizeof(dot_t));
    memcpy(l_tens,
        &(dot_t){ .a=d->a, .b=d->b, .st=d->st, .end=mid },
        sizeof(dot_t));
    dot_t* r_tens = malloc(sizeof(dot_t));
    memcpy(r_tens,
        &(dot_t){ .a=d->a, .b=d->b, .st=mid+1, .end=d->end },
        sizeof(dot_t));
   // pthread_cond_broadcast(&threads->cv);

    pthread_create(&thread1, NULL, dot_helper, l_tens);
    pthread_create(&thread2, NULL, dot_helper, r_tens);
    void* res1;
    void* res2;
    int ret = pthread_join(thread1, &res1);
    int ret2 = pthread_join(thread2, &res2);

    //Aggregate step
    if(ret != 0 || ret2 != 0) { return NULL; }
    *result = *(float *)res1 + *(float *)res2;
    free(l_tens);
    free(r_tens);
    return result;
}
float dot(const Tensor* a, const Tensor* b) {
    printf("Doing a cheeky little dot product \n");
    float dot = 0;
    if(a->dSize != b->dSize) {
        return NAN;
    }
    dot_t args = { a, b, 0, a->dSize };
    void* dot_result = dot_helper(&args);
    float ans = *(float*) dot_result;
    free(dot_result);
    return ans;
}

Tensor* matmul(const Tensor* a, const Tensor* b) {
    int shape[2] = {a->shape[0], b->shape[1]};
    Tensor* mat = create_tensor(shape, 2);
    if(a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0] ) {
        return NULL;
    }
    //foolish matrix multiplication
    for(int r = 0; r < a->shape[0]; r++) {
        for(int c = 0; c < a->shape[1]; c++) {

        }
    }
    return create_tensor(a->shape, a->ndim);
}

float item(const Tensor* t, int i) {
    if(i < 0 || i >= t->dSize) {
        return NAN;
    }
    return t->data[i];
}

float at(const int* shape, const int ndim) {
    return 0.0;
}

Tensor* equal(const Tensor* a, const Tensor* b) {
    int shape[2] = {a->shape[0], b->shape[1]};
    Tensor* mat = create_tensor(shape, 2);
    return mat; 
}