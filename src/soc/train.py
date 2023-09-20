import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torch
from model import PINN_Model
from soc.model import PINN_Model
import scipy.io as sio
import logging
import sys
import os
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from dataset.dataset import CustomDataset

def train():
    # train_inputs, train_outputs, test_inputs, test_outputs= CustomDataset().prepare_sandia_time_series(file="data/Sandia/time_series/SNL_18650_LFP_25C_0-100_0.5-1C_a_timeseries.csv", train_split=0.001, validation_split=0)
    # physics_inputs, physics_outputs, trash, trash = CustomDataset().prepare_sandia_time_series(file="data/Sandia/time_series/SNL_18650_LFP_25C_0-100_0.5-1C_b_timeseries.csv", train_split=1, validation_split=0)
    # trash, trash, test_inputs, test_outputs = CustomDataset().prepare_sandia_time_series(file="data/Sandia/time_series/SNL_18650_LFP_25C_20-80_0.5-0.5C_a_timeseries.csv", train_split=0.9999, validation_split=0)
    train_inputs, train_outputs, physics_inputs, trash = CustomDataset().prepare_sandia_time_series(file="data/Sandia/train.csv", train_split=0.8, validation_split=0.2)
    test_inputs, test_outputs, trash, trash = CustomDataset().prepare_sandia_time_series(file="data/Sandia/test.csv", train_split=1, validation_split=0)
    torch.manual_seed(0)
    # Set the seed for reproducibility
    # Create the model
    model = PINN_Model()
    # Define the initial learning rate
    initial_lr = 0.0001
    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    # Create a learning rate scheduler
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=500, min_lr=1e-6)
    # Initialize other training parameters
    best_loss = float("inf")
    # Number of epochs to wait before stopping if validation loss increases
    patience = 80000000 
    for epoch in range(200000):
        if epoch % 10 == 0:
            # Calculate the validation loss
            validation_loss = model.validation_loss(test_inputs, test_outputs).item()
            logging.info("Epoch: %d, Validation loss: %f, Best Validation loss: %f" % (epoch, validation_loss, best_loss))
            # Update the learning rate scheduler based on the validation loss
            # scheduler.step(validation_loss)
            # Check if validation loss is increasing
            if validation_loss < best_loss:
                best_loss = validation_loss
                logging.debug("Best validation loss: %f" % best_loss)
                # Save the model weights
                torch.save(model.state_dict(), "pth_models/best_model.pth")
            else:
                # If validation loss increases for 'patience' epochs, stop training
                if epoch > patience and validation_loss >= best_loss:
                    logging.info("Best validation loss: %f" % best_loss)
                    logging.info("Early stopping at epoch %d" % epoch)
                    break
        # Reset the gradients
        optimizer.zero_grad()
        # Calculate the loss
        loss = model.loss(train_inputs, train_outputs, physics_informed=False, physics_x=physics_inputs, capacity=1.1)
        if (epoch % 100 == 0):
            # Forward pass
            outputs = model(train_inputs)
            # Plot the outputs and the targets
            plt.plot(train_inputs[:, 0].detach().numpy(), outputs.detach().numpy(), label="Predicted")
            plt.plot(train_inputs[:, 0].detach().numpy(), train_outputs.detach().numpy(), label="Target")
            plt.legend()
            plt.savefig("plots/epoch_%d.png" % epoch)
            plt.close()
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
    # Calculate the loss
    loss = model.validation_loss(test_inputs, test_outputs)
    # Log the loss
    logging.info("Test loss: %f" % loss.item())
    # Save the model
    torch.save(model.state_dict(), "pth_models/model_soc_pinn.pth")


def setup_logging(level=logging.INFO):
    """
    Setup the logging environment
    """
    # Create the logger
    logger = logging.getLogger()
    # Set the log level
    logger.setLevel(level)
    # Create the formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Create the file handler
    file_handler = logging.FileHandler("log/random_tests.log")
    # Set the formatter
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)

if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    train()
    


