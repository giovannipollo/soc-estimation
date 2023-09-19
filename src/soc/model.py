import torch.nn as nn
import torch
import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from soc_ocv.model_soc_ocv import Model_Soc_Ocv

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
    def __init__(self, input_size=4, output_size=1, hidden_size=16):
        """
        Constructor of the model.
        """
        super(PINN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

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

    def loss(self, x, y, physics_x = None, physics_informed=False, capacity = 0):
        """
        Loss function of the model.
        """
        # Loss driven by data
        data_loss = nn.functional.mse_loss(self.forward(x), y)
        # Loss driven by physics
        if physics_informed:
            physics_loss = self.physics_loss_soc_de(physics_x, capacity=capacity)
        else:
            physics_loss = 0
        
        # Weights for each contribution of the loss
        data_weight = 1
        physics_weight = 1
        # Log the losses
        logging.debug("Data loss: " + str(data_loss))
        logging.debug("Physics loss: " + str(physics_loss))
        # Return the weighted sum of the losses
        return data_weight*data_loss + physics_weight*physics_loss
    
    def validation_loss(self, x, y):
        """
        Loss function of the model.
        """
        output = self.forward(x)
        return nn.functional.mse_loss(output, y)
    
    def physics_loss_Rint(self, x, y):
        """
        Equation loss of the model. The equivalent circuit is a simple resistor in series with a voltage source.
        """
        # Extract the inputs
        time_step = x[:, 0]
        voltage = x[:, 1]
        current = x[:, 2]
        temperature = x[:, 3]
        # Load the open circuit voltage model
        open_circuit_voltage_model = Model_Soc_Ocv()
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
    
    def physics_loss_soc_de(self, x, capacity):
        """
        Equation loss of the model. The equation is the following:
        dSoC/dt = -I / (3600 * C)
        where:
            - dSoC/dt: Rate of change of the SoC
            - I: Current
            - C: Capacity in Ah
        """
        # Define the inputs for the equation by selecting 10 random samples from x
        time_step = torch.tensor(x[:, 0], requires_grad=True)
        voltage = torch.tensor(x[:, 1], requires_grad=True)
        current = torch.tensor(x[:, 2], requires_grad=True)
        temperature = torch.tensor(x[:, 3], requires_grad=True)
        # Define the inputs
        physics_input = torch.stack((time_step, voltage, current, temperature), dim=1)
        # Compute the estimated SoC
        estimated_soc = self.forward(physics_input)
        estimated_soc = torch.flatten(estimated_soc)
        logging.debug("Estimated soc: ", estimated_soc)
        # Compute the derivative of the SoC with respect to time_step
        d_soc_dt = torch.autograd.grad(estimated_soc, time_step, grad_outputs=torch.ones_like(time_step), retain_graph=True, create_graph=True)[0]
        # Compute the equation loss
        logging.debug("d_soc_dt: ", d_soc_dt)
        eq_loss = torch.abs(d_soc_dt - current / (3600 * capacity))
        eq_loss = torch.mean(eq_loss)
        logging.debug("Eq loss: ", eq_loss)
        return eq_loss