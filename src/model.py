import torch.nn as nn

class Model(nn.Module):
    """
    Model class. The model must have 3 inputs and 1 output.
    The inputs are:
        - Voltage: Voltage of the battery
        - Current: Current of the battery
        - Temperature: Temperature of the battery
    
    The output is:
        - State of Charge (SoC): The SoC is a value between 0 and 1 that indicates the current capacity of the battery.
    """
    def __init__(self):
        """
        Constructor of the model.
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        output = self.fc1(x)
        output = nn.functional.relu(output)
        output = self.fc2(output)
        output = nn.functional.relu(output)
        output = self.fc3(output)
        output = nn.functional.relu(output)
        output = self.fc4(output)
        return output
