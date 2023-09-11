import torch.nn as nn
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from soc_ocv.model_soc_ocv import Model

class PINN_Model(nn.Module):
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
        super(PINN_Model, self).__init__()
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

    def loss(self, x, y):
        """
        Loss function of the model.
        """
        data_loss = nn.functional.mse_loss(self.forward(x), y)
        physics_loss = self.physics_loss(x, y)
        return data_loss + physics_loss
    
    def validation_loss(self, x, y):
        """
        Loss function of the model.
        """
        output = self.forward(x)
        return nn.functional.mse_loss(output, y)
    
    def physics_loss(self, x, y):
        """
        Equation loss of the model. The equivalent circuit is a simple resistor in series with a voltage source.
        """
        # Extract the inputs
        voltage = x[:, 0]
        current = x[:, 1]
        temperature = x[:, 2]
        # Load the open circuit voltage model
        open_circuit_voltage_model = Model()
        open_circuit_voltage_model.load_state_dict(torch.load("pth_models/model_soc_ocv.pth")) 
        # Compute the estimated SoC
        # estimated_soc = self.forward(x)
        input_tensor = torch.stack((temperature, torch.flatten(y)), dim=1)
        # Compute the open circuit voltage based on the estimated SoC and the temperature
        open_circuit_voltage_model.eval()
        open_circuit_voltage = open_circuit_voltage_model(input_tensor)
        resistence = 15e-3
        # Compute the equation loss
        eq_loss = torch.mean(torch.abs(open_circuit_voltage - voltage + resistence * current))
        # Square 
        return eq_loss