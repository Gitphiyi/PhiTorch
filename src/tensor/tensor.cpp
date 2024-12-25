#include<iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <exception>
#include <string>
#include<vector>
#include "Tensor.h"

using namespace std;

template <typename A> 
Tensor<A>::Tensor(const vector<A>& data, const vector<int>& shape, const string& device){
    int n = data.size();
    int sn = shape.size();
    int totalSize = 1;
    vector<int> stride(sn);
    if(n == 0 || sn == 0) {
        throw runtime_error("no data to put in tensor!");
    }
    for(int i = 0; i < sn; i++) {
        totalSize *= shape[i];
    }
    if(totalSize != n) {
        throw runtime_error("shape size is not same as data size");
    }
    stride[sn-1] = 1;
    int prev = shape[sn-1];
    for(int i = sn-2; i >= 0; i--) {
        stride[i] = stride[i+1] * prev;
        prev = shape[i];
    }

    this->ndim = shape.size();
    this->device = device;
    this->data = data;
    this->shape = shape;
    this->stride = stride;
    this->dSize = n;
}

template <typename A> 
Tensor<A>::~Tensor() {
    //cout << "destroyed tensor\n";
}

template <typename A> 
void Tensor<A>::print() const {
    cout<<"Data: "<<endl<<"[";
    for(int i = 0; i < dSize; i++) {
        cout << data[i];
        if(i < dSize-1) { cout << ", "; }
    }
    cout<<"]"<<endl<<"Shape: [";
    for(int i = 0; i < ndim; i++) {
        cout << shape[i];
        if(i < ndim-1) { cout << ", "; }
    }
    cout<<"]"<<endl<<"Stride: [";
    for(int i = 0; i < ndim; i++) {
        cout << stride[i];
        if(i < ndim-1) { cout << ", "; }
    }
    cout<<"]"<<endl;
}

template <typename A> 
void Tensor<A>::flatten() {
    int n = data.size();
    shape = {n};
}

template <typename A> 
void Tensor<A>::reshape(const vector<int>& shape, const int dSize) {
    if(shape.size() != ndim || this->dSize != dSize) {
        throw runtime_error("shape or data size is not the same");
    }
    this->shape = shape;
}

template <typename A> 
void Tensor<A>::transpose() {
    if(ndim == 2) {
        int temp = shape[0];
        shape[0] = shape[1];
        shape[1] = shape[0];
    }
    else {
        throw runtime_error("shape cannot be transposed as it is not 2 dimensional");
    }
}

template <typename A> 
Tensor<A> Tensor<A>::dot(const Tensor& o) const {
    if(o.dSize != dSize) {
        throw runtime_error("data size is not the same");
    }
    if(o.ndim != 1 || ndim != 1) {
        throw runtime_error("not a vector");
    }
    float val = 0;
    for(int i = 0; i < dSize; i++) {
        val += o.data[i] * data[i];
    }
    return Tensor({val}, {1}, device);
}

template <typename A> 
A Tensor<A>::item() {
    if(dSize != 1) {
        throw runtime_error("not a type tensor[0]");
    }
    return data[0];
}
template <typename A> 
A& Tensor<A>::at(const vector<int>& idx) {
    int n = idx.size();
    if(ndim != n) {
        throw runtime_error("index not correct dimensions");
    }
    int pos = 0;
    for(int i = 0; i < n; i++) {
        pos += stride[i]*idx[i];
    }
    return data[pos];
}

template <typename A> 
Tensor<A> Tensor<A>::operator+(const Tensor& o) const {
    if(o.dSize != dSize) {
        throw runtime_error("data sizes are not the same");
    }
    vector<A> temp;
    for(int i = 0; i < dSize; i++) {
        temp.push_back(o.data[i] + data[i]);
    }
    return Tensor(temp, shape, device);
}

template <typename A> 
Tensor<A> Tensor<A>::operator-(const Tensor& o) const {
    if(o.dSize != dSize) {
        throw runtime_error("data sizes are not the same");
    }
    vector<A> temp;
    for(int i = 0; i < dSize; i++) {
        temp.push_back(data[i]-o.data[i]);
    }
    return Tensor(temp, shape, device);
}

template <typename A> 
Tensor<A> Tensor<A>::operator[](int idx) const {
    int begin = stride[0]*idx;
    int end = stride[0]*(idx+1);
    vector<A> newData;
    for(int i = begin; i < end; i++) {
        newData.push_back(data[i]);
    }
    vector<int> newShape(shape.begin()+1, shape.end());
    
    if(newShape.size() == 0) {
        newShape = {1};
    }
    return Tensor(newData, newShape, device);
}

template <typename A> 
Tensor<A> Tensor<A>::zeros(vector<int>& shape, string& device) {
    int n = 1;
    for(int i = 0; i < shape.size(); i++) {
        n*=shape[i];
    }
    return Tensor(vector<A>(n, 0), shape, device);
}

template <typename A> 
Tensor<A> Tensor<A>::ones(vector<int>& shape, string& device) {
    int n = 1;
    for(int i = 0; i < shape.size(); i++) {
        n*=shape[i];
    }
    return Tensor(vector<A>(n, 1), shape, device);
}

int main() {
    vector<int> data;
    for(int i = 0; i < 48; i++) {
        data.push_back(i);
    }
    vector<int> shape = {3, 4, 4};
    string device = "cpu";

    Tensor tensor(data, shape, device);
    Tensor a = Tensor<int>::zeros(shape, device);
    tensor.print();
    //tensor.print();
    // Tensor a({1,2,3}, {3}, device);
    // Tensor b({1,2,3}, {3}, device);
    // Tensor t = a.dot(b);
    // t.print();
    return 0;
}