#include<iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <exception>
#include <string>
#include<vector>
#include "Tensor.h"

using namespace std;

Tensor::Tensor(const vector<float>& data, const vector<int>& shape, const string& device){
    int n = data.size();
    int totalSize = 0;
    for(int i = 0; i < shape.size(); i++) {
        totalSize += shape[i];
    }
    if(totalSize != n) {
        throw runtime_error("shape size is not same as data size");
    }
    this->ndim = shape.size();
    this->device = device;
    this->data = data;
    this->shape = shape;
    this->dSize = n;
}

Tensor::~Tensor() {
    cout << "destroyed tensor\n";
}

void Tensor::print() {
    for(int i = 0; i < data.size(); i++) {
        cout << data[i]<<", ";
    }
    cout<<endl;
}

void Tensor::flatten() {
    int n = data.size();
    shape = {n};
}

void Tensor::reshape(const vector<int>& shape, const int dSize) {
    if(shape.size() != ndim || this->dSize != dSize) {
        throw runtime_error("shape or data size is not the same");
    }
    this->shape = shape;
}
void Tensor::transpose() {
    if(ndim == 2) {
        int temp = shape[0];
        shape[0] = shape[1];
        shape[1] = shape[0];
    }
    else {
        throw runtime_error("shape cannot be transposed as it is not 2 dimensional");
    }
}

Tensor Tensor::operator+(const Tensor& o) {
    if(o.dSize != dSize) {
        throw runtime_error("data sizes are not the same");
    }
    vector<float> temp;
    for(int i = 0; i < dSize; i++) {
        temp.push_back(o.data[i] + data[i]);
    }
    return Tensor(temp, shape, device);
}
Tensor Tensor::operator-(const Tensor& o) {
    if(o.dSize != dSize) {
        throw runtime_error("data sizes are not the same");
    }
    vector<float> temp;
    for(int i = 0; i < dSize; i++) {
        temp.push_back(data[i]-o.data[i]);
    }
    return Tensor(temp, shape, device);
}

int main() {
    vector<float> data = {1.0, 2.0, 3.0, 4.0};
    vector<int> shape = {2, 2};
    string device = "cpu";

    Tensor tensor(data, shape, device);
    tensor.print();
    return 0;
}