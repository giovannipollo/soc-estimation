import torch.nn as nn
import torch
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from soc_ocv.model_soc_ocv import Model_Soc_Ocv


class PINN_Model(nn.Module):
    """
    Model class. The model must have 4 inputs and 1 output.
    The inputs are:
        - Time step: Time step of the battery
        - Voltage: Voltage of the battery
        - Current: Current of the battery
        - Temperature: Temperature of the battery

    The output is:
        - State of Charge (SoC): The SoC is a value between 0 and 1 that indicates the current capacity of the battery.
    """

    def __init__(self, input_size=4, output_size=1, hidden_size=16):
        """
        Constructor of the model

        Parameters
        ----------
        input_size : int, optional
            Number of inputs of the model. The default is 4.
        output_size : int, optional
            Number of outputs of the model. The default is 1.
        hidden_size : int, optional
            Number of neurons in the hidden layers. The default is 16.

        Returns
        -------
        None.
        """
        super(PINN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the model.
        """
        output = self.fc1(x)
        output = self.tanh(output)
        output = self.fc2(output)
        output = self.tanh(output)
        output = self.fc3(output)
        output = self.tanh(output)
        output = self.fc4(output)
        return output

    def loss(
        self, x, y, physics_x=None, physics_informed=False, pinn_type="cc", capacity=0
    ):
        """
        Loss function of the model.

        Parameters
        ----------
        x : torch.tensor
            Inputs of the model.
        y : torch.tensor
            Outputs of the model.
        physics_x : torch.tensor, optional
            Inputs of the model for the physics loss. The default is None.
        physics_informed : bool, optional
            If True, the physics loss is computed. The default is False.
        pinn_type : str, optional
            Type of the physics loss. The default is None. Allowed values are:
                - "cc": Coulomb counting
                - "rint": Resistor in series with a voltage source
        capacity : float, optional
            Nominal capacity of the battery. The default is 0.

        Returns
        -------
        loss : float
            Loss of the model computed summing the mean squared error and the physics loss.
        """
        # Loss driven by data
        data_loss_function = nn.MSELoss()
        data_loss = data_loss_function(self.forward(x), y)
        # Loss driven by physics
        if physics_informed:
            if pinn_type == "cc":
                physics_loss = self.physics_loss_soc_de(physics_x, capacity=capacity)
            elif pinn_type == "rint":
                physics_loss = self.physics_loss_Rint(physics_x, y)
            else:
                physics_loss = 0
        else:
            physics_loss = 0

        # Weights for each contribution of the loss
        data_weight = 1
        physics_weight = 0.0001
        # Log the losses
        logging.info("Data loss: " + str(data_loss))
        logging.info("Physics loss: " + str(physics_loss))
        # Return the weighted sum of the losses
        return data_weight * data_loss + physics_weight * physics_loss

    def validation_loss(self, x, y):
        """
        Loss function of the model.

        Parameters
        ----------
        x : torch.tensor
            Inputs of the model.
        y : torch.tensor
            Outputs of the model.

        Returns
        -------
        loss : float
            Loss of the model computed using the mean squared error.
        """
        output = self.forward(x)
        validation_loss_function = nn.MSELoss()
        return validation_loss_function(output, y)

    def physics_loss_Rint(self, x, y):
        """
        Equation loss of the model. The equivalent circuit is a simple resistor in series with a voltage source.

        Parameters
        ----------
        x : torch.tensor
            Inputs of the model.
        y : torch.tensor
            Outputs of the model.

        Returns
        -------
        eq_loss : float
            Equation loss of the model.
        """
        # Extract the inputs
        voltage = x[:, 1]
        current = x[:, 2]
        temperature = x[:, 3]
        # Load the open circuit voltage model
        open_circuit_voltage_model = Model_Soc_Ocv()
        open_circuit_voltage_model.load_state_dict(
            torch.load("pth_models/model_soc_ocv.pth")
        )
        # Compute the estimated SoC
        input_tensor = torch.stack((temperature, torch.flatten(y)), dim=1)
        # Compute the open circuit voltage based on the estimated SoC and the temperature
        open_circuit_voltage_model.eval()
        open_circuit_voltage = open_circuit_voltage_model(input_tensor)
        resistence = 15e-3
        # Compute the equation loss
        eq_loss = torch.mean(
            torch.abs(open_circuit_voltage - voltage + resistence * current)
        )
        return eq_loss

    def physics_loss_soc_de(self, x, capacity):
        """
        Equation loss of the model. The equation is the following:
        dSoC/dt = I / C
        where:
            - dSoC/dt: Rate of change of the SoC
            - I: Current in Ah
            - C: Nominal Capacity of the battery in Ah

        Parameters
        ----------
        x : torch.tensor
            Inputs of the model.
        capacity : float
            Nominal capacity of the battery.

        Returns
        -------
        eq_loss : float
            Equation loss of the model.

        """
        # Define the inputs for the equation by selecting 10 random samples from x
        time_step = x[:, 0].clone().detach().requires_grad_(True)
        voltage = x[:, 1].clone().detach().requires_grad_(True)
        current = x[:, 2].clone().detach().requires_grad_(True)
        temperature = x[:, 3].clone().detach().requires_grad_(True)
        # Define the physics inputs
        physics_input = torch.stack((time_step, voltage, current, temperature), dim=1)
        # Compute the estimated SoC
        estimated_soc = self.forward(physics_input)
        estimated_soc = torch.flatten(estimated_soc)
        logging.debug("Estimated soc: ", estimated_soc)
        # Compute the derivative of the SoC with respect to time_step
        d_soc_dt = torch.autograd.grad(
            estimated_soc,
            time_step,
            grad_outputs=torch.ones_like(time_step),
            create_graph=True,
        )[0]
        # Compute the equation loss
        logging.debug("d_soc_dt: ", d_soc_dt)
        eq_loss_function = nn.MSELoss()
        eq_loss = eq_loss_function(d_soc_dt, current / capacity)
        return eq_loss

    def plot_epoch_loss(self, train_loss, validation_loss, epoch):
        """
        Plot the loss for each epoch

        Parameters
        ----------
        train_loss : list
            List containing the training loss for each epoch.
        validation_loss : list
            List containing the validation loss for each epoch.
        epoch : int
            Number of epochs.

        Returns
        -------
        None.
        """
        # Plot the loss
        plt.figure()
        plt.plot(np.arange(epoch), train_loss, label="Train loss")
        plt.plot(np.arange(epoch), validation_loss, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("loss.png")
        plt.close()
