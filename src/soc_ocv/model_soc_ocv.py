import torch.nn as nn

class Model(nn.Module):
    """
    Model class. The model must have 2 inputs and 1 output.
    The inputs are:
        - Temperature: Temperature of the battery
        - SoC: State of Charge of the batter
    The output is:
        - Open Circuit Voltage (OCV)
    """
    def __init__(self):
        """
        Constructor of the model.
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 1)

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
        output = nn.functional.relu(output)
        output = self.fc5(output)
        output = nn.functional.relu(output)
        output = self.fc6(output)
        return output
    
    def loss(self, x, y):
        """
        Loss function of the model.
        """
        output = self.forward(x)
        return nn.functional.mse_loss(output, y)
