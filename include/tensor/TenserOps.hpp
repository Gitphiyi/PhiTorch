#pragma once

#include "Tensor.hpp"

Tensor* sigmoid(const Tensor* t);
Tensor* relu(const Tensor* t);

void add(Tensor* out, Tensor* a, Tensor* b);