#pragma once
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include<vector>
#include "../tensor/Tensor.h"

class Linear {
    private:
        Tensor<float> weight;
        Tensor<float> bias; 
    public:
        Linear(int in_features, int out_features, bool bias);
};