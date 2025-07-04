#include <stdio.h>
#include <stdlib.h>
#include "tensor/Tensor.h"

int main() {
    printf("Hello world! \n");
    //int * a =(int*)malloc(s*s*sizeof(int));
    int shape[2] = {2,20};
    int ndim = 2;
    float a_data[40];
    float b_data[40]; //5 + 25+25+36 = 91
    for(int i = 0; i < 40; i++) {
        a_data[i] = (float) rand();
        b_data[i] = (float) rand();
    }

    Tensor* a = create_tensor(shape, ndim);
    Tensor* b = create_tensor(shape, ndim);
    set_data(a, a_data, 40);
    set_data(b, b_data, 40);
    float actual_dot = 0;
    for(int i = 0; i < a->dSize; i++) {
        actual_dot += a->data[i] * b->data[i];
    }
    float res = dot(a,b);
    printf("End dot prod; %f\n", res);
    printf("Real dot prod; %f\n", actual_dot);

    //print_metadata(a);
    //delete_tensor(a);
    return 0;
}
