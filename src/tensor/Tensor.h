#pragma once
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include<vector>

using namespace std;
/*
Resources used:
- https://towardsdatascience.com/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc
 */
class Tensor {
    private:
        vector<float> data; // array of data
        vector<int> shape; // array of shape of each dimension 
        vector<int> stride; // indices needed to traverse to get to a certain index. i.e. shape=[3,4,4] then stride = [16,4,1]
        int ndim; // number of dimensions (rank)
        int dSize; // size of data
        string device; //cpu/gpu

    public:
        Tensor(const vector<float>& data, const vector<int>& shape, const string& device);
        ~Tensor();

        void print() const;
        void flatten(); //collapse dimension into 1
        void reshape(const vector<int>& shape, const int dSize);
        void transpose();
        Tensor dot(const Tensor& o) const;
        Tensor matmul(Tensor& o);
        float item(); //gets element of tensor[1]
        float& at(const vector<int>& idx);

        Tensor operator=(const Tensor& o);
        Tensor operator+(const Tensor& o) const;
        Tensor operator-(const Tensor& o) const;
        Tensor operator[](int idx) const; // has to be read-only due to being c++


        static Tensor zeros(vector<int>& shape, string& device);  
        static Tensor ones(vector<int>& shape, string& device);
        static Tensor rand(vector<int>& shape);
        static Tensor eye(vector<int>& shape);

};