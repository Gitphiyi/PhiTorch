#include "tensor/Tensor.hpp"
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

#include <cstring> 
#include <cstdlib> 
#include <algorithm>

using namespace std;

Tensor::Tensor(int* shape_, int ndim_, const char* device_) : ndim(ndim_), device(device_) {
        dSize = 1;
        stride = new int[ndim];
        shape = new int[ndim];
        memcpy(shape, shape_, sizeof(int) * ndim);

        for(int i = ndim-1; i >= 0; i--) {
            dSize *= shape[i];
            if( i < ndim-1) {
                stride[i] = stride[i+1] * shape[i+1];
            } 
            else {
                stride[i] = 1;
            }
        }
        if(ndim == 1) {
            stride[0] = shape[0];
            dSize = shape[0];
        }

        data = new float[dSize];
        grad = new float[dSize];
        fill(data, data + dSize, 0.0f);
        fill(grad, grad + dSize, 0.0f);

        //memcpy device string as well as it will dangle once it is over
};
Tensor::~Tensor() {
    delete[] data;
    delete[] grad;
    delete[] shape;
    delete[] stride;
}

int Tensor::set_data(float* newData, int size) {
    if( size != dSize) {
        printf("dSize: %d != size: %d\n", dSize, size);
        return -1;
    }
    for(int i = 0; i < size; i ++) {
        data[i] = newData[i];
    }
    return 0;
}
void Tensor::print_metadata(int max_print) {
    printf("-----------------Tensor Print-----------------\n");
    printf("%d-dim %s Tensor: size = %d \n", ndim, device, dSize);
    printf("Stride = [");
    for(int i = 0; i < ndim; i++) {
        printf("%d", stride[i]);
        if(i < ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("Shape = [");
    for(int i = 0; i < ndim; i++) {
        printf("%d", shape[i]);
        if(i < ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("Data = [");
    int end = std::min(max_print, dSize);
    for(int i = 0; i < end; i++) {
        printf("%.2f", data[i]);
        if(i < end - 1) {
            printf(", ");
        }
    }
    if(end < dSize) {
        printf(",..., %.2f", data[dSize-1]);
    }
    printf("]\n");
}

Tensor* Tensor::flatten() {
    int new_shape[1] = { dSize };
    Tensor* t = new Tensor(new_shape, 1, device);
    if (data && t->data) {
        memcpy(t->data, data, sizeof(float) * dSize);
    }
    if (grad && t->grad) {
        memcpy(t->grad, grad, sizeof(float) * dSize);
    }
    return t;
}

Tensor* Tensor::reshape(int* new_shape, int new_dim) {
    Tensor* t = new Tensor(new_shape, new_dim, device);
    if (data && t->data) {
        memcpy(t->data, data, sizeof(float) * dSize);
    }
    if (grad && t->grad) {
        memcpy(t->grad, grad, sizeof(float) * dSize);
    }
    return t;
}

Tensor* Tensor::transpose() {
    if(ndim != 2) {
        printf("Not a 2-D Tensor");
        return NULL;
    }

    const int rows = shape[0];
    const int cols = shape[1];
    int new_shape[2] = { cols, rows };
    Tensor* out = new Tensor(new_shape, 2, device);

    if(strcmp(device, GPU) == 0) {
        printf("is a gpu \n");
        return nullptr;
    }

    for (int i = 0; i < rows; ++i) {
        const int in_row_base = i * cols;  
        for (int j = 0; j < cols; ++j) {
            const int src = in_row_base + j; 
            const int dst = j * rows + i;   
            if (data) out->data[dst] = data[src];
            if (grad) out->grad[dst] = grad[src];
        }
    }
    return out;
    //multithread this if bigger than cache line
}

// /**
//  * Dot product process can be split into two steps: 
//  * Multiplying pairwise a and b values
//  * Summing the products into one value
//  * 
//  */
// typedef struct dot_t {
//     const Tensor* a;
//     const Tensor* b;
//     const int st;
//     const int end;
// } dot_t;
// void* dot_helper(void* args) {
//     dot_t* d = (dot_t*) args;
//     const int maxLoopIter = 5;
//     const int mid = (d->end + d->st) / 2;
//     float* result = (float*) malloc(sizeof(float));
//     *result = 0.0;
//     //Base Case
//     if (d->end - d->st <= maxLoopIter) {
//         for(int i = d->st; i <= d->end; i++) {
//             printf("num %d: %f\n", i, d->a->data[i]);
//             *result += d->a->data[i] * d->b->data[i];
//         }
//         return result;
//     }

//     //Thread step
//     pthread_t thread1;
//     pthread_t thread2;
//     dot_t* l_tens = (dot_t*) malloc(sizeof(dot_t));
//     memcpy(l_tens,
//         &(dot_t){ .a=d->a, .b=d->b, .st=d->st, .end=mid },
//         sizeof(dot_t));
//     dot_t* r_tens = (dot_t*) malloc(sizeof(dot_t));
//     memcpy(r_tens,
//         &(dot_t){ .a=d->a, .b=d->b, .st=mid+1, .end=d->end },
//         sizeof(dot_t));
//    // pthread_cond_broadcast(&threads->cv);

//     pthread_create(&thread1, NULL, dot_helper, l_tens);
//     pthread_create(&thread2, NULL, dot_helper, r_tens);
//     void* res1;
//     void* res2;
//     int ret = pthread_join(thread1, &res1);
//     int ret2 = pthread_join(thread2, &res2);

//     //Aggregate step
//     if(ret != 0 || ret2 != 0) { return NULL; }
//     *result = *(float *)res1 + *(float *)res2;
//     free(l_tens);
//     free(r_tens);
//     return result;
// }
// float dot(const Tensor* a, const Tensor* b) {
//     printf("Doing a cheeky little dot product \n");
//     float dot = 0;
//     if(a->dSize != b->dSize) {
//         return NAN;
//     }
//     dot_t args = { a, b, 0, a->dSize };
//     void* dot_result = dot_helper(&args);
//     float ans = *(float*) dot_result;
//     free(dot_result);
//     return ans;
// }

// Tensor* matmul(const Tensor* a, const Tensor* b) {
//     int shape[2] = {a->shape[0], b->shape[1]};
//     Tensor mat = Tensor(shape, 2);
//     if(a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0] ) {
//         return NULL;
//     }
//     //foolish matrix multiplication
//     for(int r = 0; r < a->shape[0]; r++) {
//         for(int c = 0; c < a->shape[1]; c++) {

//         }
//     }
//     return &Tensor(a->shape, a->ndim);
// }

// float item(const Tensor* t, int i) {
//     if(i < 0 || i >= t->dSize) {
//         return NAN;
//     }
//     return t->data[i];
// }

// float at(const int* shape, const int ndim) {
//     return 0.0;
// }

// Tensor* equal(const Tensor* a, const Tensor* b) {
//     int shape[2] = {a->shape[0], b->shape[1]};
//     Tensor* mat = &Tensor(shape, 2);
//     return mat; 
// }