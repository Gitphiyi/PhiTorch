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
        //string device; //cpu/gpu

    public:
        string device; //cpu/gpu
        Tensor(const vector<float>& data, const vector<int>& shape, const string& device);
        ~Tensor();

        void print() const;
        void flatten(); //collapse dimension into 1
        void reshape(const vector<int>& shape, const int dSize);
        void transpose();
        Tensor dot(const Tensor& o) const;
        Tensor matmul(Tensor& o);
        float item(); //gets element of tensor[1]

        Tensor operator+(const Tensor& o) const;
        Tensor operator-(const Tensor& o) const;
        Tensor operator[](int idx) const; //first dimension
        //float operator[](int idx); //read-write final dimension


        Tensor zeros(vector<int>& shape);  
        Tensor ones(vector<int>& shape);
        Tensor rand(vector<int>& shape);
        Tensor eye(vector<int>& shape);

};