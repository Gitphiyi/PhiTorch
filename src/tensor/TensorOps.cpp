#include "TenserOps.hpp"

void add(Tensor* out, Tensor* a, Tensor* b) {
    //Do dSize error checking and such
    
    for(int i = 0; i < out->dSize; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}