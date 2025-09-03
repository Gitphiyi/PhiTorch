#pragma once

#include "Node.hpp"
#include "Tensor.hpp"

//typedef void (*BackwardFn)(std::vector<Node*> t);

void backwardAdd(Node* output, Node* a, Node* b) {
    if(a->requires_grad) {
        for (size_t i = 0; i < output->val->dSize; ++i)
            a->val->grad[i] += output->val->grad[i];
    }
    if(b->requires_grad) {
        for (size_t i = 0; i < output->val->dSize; ++i)
            b->val->grad[i] += output->val->grad[i];
    }
}

void backwardSub(Node* output, Node* a, Node* b) {
    if(a->requires_grad) {
        for (size_t i = 0; i < output->val->dSize; ++i)
            a->val->grad[i] += output->val->grad[i];
    }
    if(b->requires_grad) {
        for (size_t i = 0; i < output->val->dSize; ++i)
            b->val->grad[i] -= output->val->grad[i];
    }
}