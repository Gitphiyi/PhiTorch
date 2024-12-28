import nn as nn
import numpy as np


class CustomModel():
    def __init__(self):
        self.layer = nn.Linear(4, 2)
        print(self.layer)
    def forward(self, x):
        output = self.layer(x)
        return output

model = CustomModel()
input_tensor = np.array([1,1,1,1])
output = model.forward(input_tensor)
print(output)

