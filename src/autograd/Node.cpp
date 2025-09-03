#include "Node.hpp"
#include "tensor/TensorOps.hpp"

#include <assert.h>
#include <iostream>

using namespace std;

Node::Node(int dSize, const char* device, vector<Node*> children_ = {}, string op_ = "", bool requires_grad_ = false, string label_ = "") : 
            children(move(children_)),
            op(op_),
            requires_grad(requires_grad_),
            label(label_)  {
    val = new Tensor(dSize, device);
    backward = nullptr;
}
Node::~Node() {
    delete val;
};

void Node::print() {
    cout << "------------" << label << " " << op << " Node" << "------------ \n";
    cout << "Children: \n [";
    for(int i = 0; i < children.size(); i++) {
        cout << children[i]->label;
        if(i < children.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]\n";
    val->print_metadata(5);
}

Node* Node::add(Node* o) {
    assert(val->dSize != o->val->dSize && "Tensor lengths differ for add");
    vector<Node*> children_ = {this, o};
    Node* output = new Node(val->dSize, val->device, children, "+", true);
    output->backward = &backwardAdd;
    add(output->val, this->val, o->val);
    return output;
}