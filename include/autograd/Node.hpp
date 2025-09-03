#pragma once
#include <memory>
#include <vector>

struct Tensor; // forward declare

namespace ag {

struct Edge {
    Node* node; //parent node
    Tensor* tensor; // pointer to the parent tensor
};

// Base class for all autograd nodes
struct Node : std::enable_shared_from_this<Node> {
    std::vector<Edge> parents; // inputs that produced this node's output

    virtual ~Node() = default;

    // Backward function: given grad_output from the next layer,
    // compute gradients wrt each parent and accumulate into their .grad.
    virtual void backward(const Tensor& grad_output) = 0;
};

}
