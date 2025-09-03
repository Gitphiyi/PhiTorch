#pragma once
#include "Tensor.hpp"
#include "BackwardFns.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <string>

struct Node;

typedef void (*BackwardFn)(Node* output, Node* a, Node* b);

struct Node {
    Node(int dSize_, const char* device_, std::vector<Node*> children_ = {}, std::string op_ = "", bool requires_grad_ = false, std::string label_ = "");
    //Node(int dSize_, const char* device_, std::vector<Node*> children_ = {}, std::string op_ = "", bool requires_grad_ = false, std::string label_ = "");
    ~Node();
    void print();
    Node* add(Node* o);
    Node* sub(Node* o);
    Node* mul(Node* o);

    BackwardFn              backward;
    std::string             op;
    std::string             label;
    Tensor*                 val;
    std::vector<Node*>      children;
    bool                    requires_grad;
};

