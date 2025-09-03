#pragma once

inline constexpr const char* CPU = "cpu";
inline constexpr const char* GPU = "gpu";

struct Tensor {
    float* data;
    float* grad;    
    int* shape;     //shape of each dim
    int* stride;    //(Row Major) indices needed to traverse to get to 1st index. i.e. shape=[3,4,4] then stride = [16,4,1]
    int ndim;       //rank
    int dSize;      //size of data
    const char* device;   //cpu/gpu

    Tensor(int* shape_, int ndim_, const char* device); 
    Tensor(int dSize_, const char* device);
    //implement Tensors as lazily allocated objects later
    ~Tensor();
    int     set_data(float* newData, int size);
    void    print_metadata(int max_print);
    Tensor* flatten(); //collapse dimension into 1
    Tensor* reshape(int* new_shape, int new_dim);
    Tensor* transpose(); //only allow for 2D Tensors
    float   item(int i); //gets element of tensor[1]
    float   at(const int* indices, int ndim);

    bool operator==(const Tensor* o);
    bool operator!=(const Tensor* o);
    void operator+=(const Tensor* o);
    void operator-=(const Tensor* o);
};

float   dot(const Tensor* a, const Tensor* b);
Tensor* matmul(const Tensor* a, const Tensor* b);

// Generate Tensors
Tensor* zeros(const int* shape, const char* device);  
Tensor* ones(const int* shape, const char* device);
Tensor* rand_tens(const int* shape, const int ndim);
Tensor* eye(const int* shape, const int ndim);
