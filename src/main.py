import nn as nn
import nn.Optimizer as optim
import numpy as np

""" 
    Example Neural Network that shows all basic methods that must be implemented so 
"""
class CustomModel():
    def __init__(self):
        self.layer = nn.Linear(4, 2)
        print(self.layer)
    def forward(self, x):
        output = self.layer(x)
        return output

model = CustomModel()
input_tensor = np.array([1,1,1,1])
target_tensor = np.array([2,2,2,2])
output = model.forward(input_tensor)

criterion = nn.MSELoss()
optimizer = optim.SGD()
for epoch in range(10000):
    #forward pass
    output = model.forward(input_tensor)
    loss = criterion(output, target_tensor)
    #backward pass
    optimizer.zero_grad()
    loss.backward() 
    #update weights
    optimizer.step()
    
# Now model is trained with the correct 
