typedef struct Tensor {
    float* data;
    float* grad;    
    int* shape;     //shape of each dim
    int* stride;    //indices needed to traverse to get to a certain index. i.e. shape=[3,4,4] then stride = [16,4,1]
    int ndim;       //rank
    int dSize;      //size of data
    const char* device;   //cpu/gpu

    //Tensor Functions
    void (*print)(Tensor*);
} Tensor;

Tensor* create_tensor(int* shape, int ndim);
void    delete_tensor(Tensor* t);

void    print(Tensor* t);
Tensor* flatten(Tensor* t); //collapse dimension into 1
Tensor* reshape(const int* shape, const int ndim, const int dSize);
Tensor* transpose(); //only allow for 2D Tensors
float   dot(const Tensor* o);
Tensor* matmul(const Tensor* a, const Tensor* b);
float   item(); //gets element of tensor[1]
float   at(const int* idx, const int ndim);

Tensor* equal(const Tensor* t);
Tensor* add(const Tensor* a, const Tensor* b);
Tensor* sub(const Tensor* a, const Tensor* b);

Tensor* zeros(const int* shape, const char* device);  
Tensor* ones(const int* shape, const char* device);
Tensor* rand(const int* shape, const int ndim);
Tensor* eye(const int* shape, const int ndim);
