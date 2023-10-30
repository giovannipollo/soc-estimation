import torch.nn as nn
import torch
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

class LSTM_Model(nn.Module):
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
            Number of neurons in the hidden layers. The default is 16
        """

        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the model.
        """
        h_0 = torch.zeros(1, x.size(0), 16)
        c_0 = torch.zeros(1, x.size(0), 16)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = self.fc(output[:, -1, :])
        output = self.tanh(output)
        return output

    def loss(self, y_pred, y_true):
        """
        Loss function of the model. The loss function is the Mean Square Error (MSE).

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return torch.mean((y_pred - y_true) ** 2)
