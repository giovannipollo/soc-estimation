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
from dataset.sandia import SandiaDataset

def train():
    # Limit the number of cores used by PyTorch to avoid using all the cores
    torch.set_num_threads(6)
    logging.debug("Loading dataset")
    dataset = SandiaDataset(
        file="data/Sandia/time_series/SNL_18650_LFP_15C_0-100_0.5-1C_a_timeseries.csv", 
        train_cycles = 1, 
        test_cycles = 1, 
        physics_cycles = 1,
        nominal_capacity=1.1,
        threshold=0.8
    )
    train_inputs, train_outputs = dataset.get_train_data()
    test_inputs, test_outputs = dataset.get_test_data()
    physics_inputs = dataset.get_physics_input()
    physics_outputs = dataset.get_physics_output()
    # Set the seed for reproducibility
    torch.manual_seed(0)
    # Create the model
    logging.debug("Creating model")
    model = PINN_Model(hidden_size=20)
    # Define the initial learning rate
    initial_lr = 0.001
    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    # Create a learning rate scheduler
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=500, min_lr=1e-6)
    # Initialize other training parameters
    best_loss = float("inf")
    # Number of epochs to wait before stopping if validation loss increases
    logging.debug("Starting training")
    patience = 80000000 
    for epoch in range(200000):
        # Plot the prediction for the train data
        # if epoch % 100 == 0:
        #     plot_epoch_predictions_train(epoch = epoch, model = model, train_inputs = train_inputs, train_outputs = train_outputs)
        if epoch % 1000 == 0:
            plot_epoch_prediction_test(epoch = epoch, model = model, test_inputs = test_inputs, test_outputs = test_outputs)
            plot_epoch_prediction_physic(epoch = epoch, model = model, physics_inputs = physics_inputs, physics_outputs = physics_outputs)
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
        loss = model.loss(train_inputs, train_outputs, physics_informed=True, physics_x=physics_inputs, capacity=1.1)
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

def plot_epoch_prediction_test(epoch, model, test_inputs, test_outputs):
    """
    Plot the prediction of the model for the test data

    Parameters
    ----------
    epoch : int
        Current epoch.
    model : PINN_Model
        Model to use for the prediction.
    test_inputs : torch.tensor
        Inputs of the test data.
    test_outputs : torch.tensor
        Outputs of the test data.

    Returns
    -------
    None.
    """
    # Plot the prediction for the test data
    plt.figure()
    plt.scatter(test_inputs[:, 0].detach().numpy(), test_outputs.detach().numpy(), label="True SoC")
    plt.scatter(test_inputs[:, 0].detach().numpy(), model.forward(test_inputs).detach().numpy(), label="Predicted SoC")
    plt.xlabel("Time (s)")
    plt.ylabel("SoC")
    plt.legend()
    plt.savefig("plots/epoch.png")
    plt.close()

def plot_epoch_predictions_train(epoch, model, train_inputs, train_outputs):
    """
    Plot the prediction of the model for the train data

    Parameters
    ----------
    epoch : int
        Current epoch.
    model : PINN_Model
        Model to use for the prediction.
    train_inputs : torch.tensor
        Inputs of the train data.
    train_outputs : torch.tensor
        Outputs of the train data.

    Returns
    -------
    None.
    """
    # Plot the prediction for the train data
    plt.figure()
    # Plot the points without the line
    plt.scatter(train_inputs[:, 0].detach().numpy(), train_outputs.detach().numpy(), label="True SoC", s=1)
    plt.scatter(train_inputs[:, 0].detach().numpy(), model.forward(train_inputs).detach().numpy(), label="Predicted SoC", s=1)
    plt.xlabel("Time (s)")
    plt.ylabel("SoC")
    plt.legend()
    plt.savefig("plots/epoch_%d_train.png" % epoch)
    plt.close()

def plot_epoch_prediction_physic(epoch, model, physics_inputs, physics_outputs):
    """
    Plot the prediction of the model for the physics data

    Parameters
    ----------
    epoch : int
        Current epoch.
    model : PINN_Model
        Model to use for the prediction.
    physics_inputs : torch.tensor
        Inputs of the physics data.

    Returns
    -------
    None.
    """
    # Plot the prediction for the physics data
    plt.figure()
    plt.scatter(physics_inputs[:, 0].detach().numpy(), model.forward(physics_inputs).detach().numpy(), label="Predicted SoC Physics", s=1)
    plt.scatter(physics_inputs[:, 0].detach().numpy(), physics_outputs.detach().numpy(), label="True SoC Physics", s=1)
    plt.scatter
    plt.xlabel("Time (s)")
    plt.ylabel("SoC")
    plt.legend()
    plt.savefig("plots/epoch_physic.png")
    plt.close()

if __name__ == "__main__":
    # Setup the logging environment withouth debug and info messages
    setup_logging(level=logging.INFO)
    train()
    


