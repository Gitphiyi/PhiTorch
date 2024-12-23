#pragma once
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include<vector>

using namespace std;

class Tensor {
    private:
        vector<float> data; // array of data
        vector<int> shape; // array of shape of each dimension 
        int ndim; // number of dimensions (rank)
        int dSize; // size of data
        string device; //cpu/gpu

    public:
        Tensor(const vector<float>& data, const vector<int>& shape, const string& device);
        ~Tensor();

        void print() const;
        void flatten(); //collapse dimension into 1
        void reshape(const vector<int>& shape, const dSize);
        void transpose();
        int dot(Tensor& o);
        Tensor matmul(Tensor& o);

        Tensor operator+(const Tensor& o) const;
        Tensor operator-(const Tensor& o) const;
        Tensor operator[](int idx); //first dimension
        &float operator[](int idx); //read-write final dimension
        const &float operator[](vector<int> idx) const; //read-only 


        Tensor zeros(vector<int>& shape);  
        Tensor ones(vector<int>& shape);
        Tensor rand(vector<int>& shape);
        Tensor eye(vector<int>& shape);
};