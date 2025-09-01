#pragma once

struct Tensor {
    float* data;
    float* grad;    
    int* shape;     //shape of each dim
    int* stride;    //(Row Major) indices needed to traverse to get to 1st index. i.e. shape=[3,4,4] then stride = [16,4,1]
    int ndim;       //rank
    int dSize;      //size of data
    const char* device;   //cpu/gpu
};

Tensor* create_tensor(int* shape, int ndim);
void    delete_tensor(Tensor* t);

int     set_data(Tensor* tens, float* data, int size);
void    print_metadata(Tensor* t);
Tensor* flatten(const Tensor* t); //collapse dimension into 1
Tensor* reshape(const Tensor* t, int* shape, int ndim);
Tensor* transpose(const Tensor* t); //only allow for 2D Tensors
float   dot(const Tensor* a, const Tensor* b);
Tensor* matmul(const Tensor* a, const Tensor* b);
float   item(const Tensor* t, int i); //gets element of tensor[1]
float   at(const int* shape, int ndim);


Tensor* zeros(const int* shape, const char* device);  
Tensor* ones(const int* shape, const char* device);
Tensor* rand_tens(const int* shape, const int ndim);
Tensor* eye(const int* shape, const int ndim);
