#include "../tensor/Tensor.h"
#include <cassert>
#include <random>

int test_dot() {
    const int SIZE = 10;
    int shape[1] = { SIZE };
    Tensor* a = create_tensor(shape, 1);
    Tensor* b = create_tensor(shape, 1);
    //***IMPLEMENT*** check for tensor dim correctness

    //checking integer only dot product
    int int_dot = 1;
    for( int i = 0; i < SIZE; i++) {
        a->data[i] = rand();
        b->data[i] = rand();
        int_dot += a->data[i] * b->data[i];
    }
    assert( int_dot == std::trunc(dot(a,b)) );
    return 1;
}

int main() {
    return 1;
}
