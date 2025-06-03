#include <stdio.h>
#include <stdlib.h>
#include "tensor/Tensor.h"

int main() {
    printf("Hello world\n");
    //int * a =(int*)malloc(s*s*sizeof(int));
    int shape[2] = {1,2};
    int ndim = 2;
    Tensor* t = create_tensor(shape, ndim);
    //t->print(t);
    delete_tensor(t);
    return 0;
}
