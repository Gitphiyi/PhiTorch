#include "tensor/Tensor.hpp"

#include <stdio.h>
#include <stdlib.h>

void test_setdata() {
    printf("\nRun setdata() \n");
    int shape[2] = {2,20};
    Tensor t = Tensor(shape, 2, "cpu");

    int sz= 40;
    float a_data[sz];
    for(int i = 0; i < sz; i++) {
        a_data[i] = ((float) rand()) / 100;
    }
    t.set_data(a_data, sz);
    t.print_metadata(3);
}

void test_flatten() {
    int shape[2] = {2,20};
    Tensor t = Tensor(shape, 2, CPU);
    Tensor* new_tens = t.flatten();
    t.print_metadata(3);
    new_tens->print_metadata(3);
}

void test_reshape() {
    int shape[2] = {2,20};
    int new_shape[4] = {2, 2, 10};
    Tensor t = Tensor(shape, 2, CPU);
    Tensor* new_tens = t.reshape(new_shape, 3);
    t.print_metadata(3);
    new_tens->print_metadata(3);
}
void test_transpose() {
    printf("\nRun transpose() \n");
    int shape[2] = {2,4};
    Tensor t = Tensor(shape, 2, CPU);
    int sz= shape[0] * shape[1];
    float data[] = {1,2,3,4,5,6,7,8};

    t.set_data(data, sz);
    t.print_metadata(10);
    Tensor* new_tens = t.transpose();
    if(new_tens == nullptr) {
        printf("nullptr\n");
    } else {
        new_tens->print_metadata(10);
    }
}
void test_dot() {
    int shape[2] = {2,20};
    Tensor a = Tensor(shape, 2, CPU);
    Tensor b = Tensor(shape, 2, "cpu");

   float actual_dot = 0;
   for(int i = 0; i < a.dSize; i++) {
       actual_dot += a.data[i] * b.data[i];
   }
   //float res = dot(&a, &b);
   //printf("End dot prod; %f\n", res);
   //printf("Real dot prod; %f\n", actual_dot);
}


int main() {
    printf("Checking all Tensor Ops\n");

    //test_setdata();
    //test_flatten();
    //test_reshape();
    test_transpose();
    return 0;
}