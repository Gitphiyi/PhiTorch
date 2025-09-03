#include "tensor/Tensor.hpp"

#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Hello world!\n");
    int * a =(int*)malloc(sizeof(int));
    int shape[2] = {2,20};
    int ndim = 2;
    float a_data[40];
    float b_data[40]; //5 + 25+25+36 = 91
    for(int i = 0; i < 40; i++) {
        a_data[i] = (float) rand();
        b_data[i] = (float) rand();
    }
    Tensor b = Tensor(shape, ndim);
    b.print_metadata();
    return 0;
}