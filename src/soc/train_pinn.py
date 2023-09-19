import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torch
from model import Model
from pinn_model import PINN_Model
import scipy.io as sio
import logging
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from dataset.dataset import CustomDataset


def single_dataset():
    train_inputs, train_outputs, test_inputs, test_outputs = CustomDataset("lg")
    # Initiaze the seed for reproducibility
    torch.manual_seed(0)
    # Create the model
    model = PINN_Model()
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Train the model
    for epoch in range(50000):
        if epoch % 10 == 0:
            # Test the model
            outputs = model(test_inputs)
            # Calculate the loss
            loss = model.validation_loss(test_inputs, test_outputs)
            # Print the loss
            print("Test loss: %f" % loss.item())
        # Reset the gradients
        optimizer.zero_grad()
        # Calculate the loss
        loss = model.loss(train_inputs, train_outputs)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Print the loss
        # print("Epoch: %d, Loss: %f" % (epoch, loss.item()))
    # Calculate the loss
    loss = model.validation_loss(test_inputs, test_outputs)
    # Print the loss
    print("Test loss: %f" % loss.item())
    # Save the model
    torch.save(model.state_dict(), "pth_models/model_soc_pinn.pth")

def double_dataset():
    # Limit the number of cpu cores to 1
    # torch.set_num_threads(16)
    # Load the csv file with the data
    data = pd.read_csv("data/549_Dis_0p5C.csv")
    # Extract the data we want to use
    data = data[["Step Time", "Voltage", "Current", "Temperature", "Capacity"]]
    # Prepare the step time
    prepare_step_time(data)
    # Load the csv file with the test data
    test_data = pd.read_csv("data/549_HPPC.csv")
    # Extract the data we want to use
    test_data = test_data[["Step Time", "Voltage", "Current", "Temperature", "Capacity"]]
    # Prepare the step time
    prepare_step_time(test_data)
    # Split the data into train and test
    train_data = data
    test_data = test_data
    # Extract the inputs and outputs
    train_inputs = train_data[["Step Time", "Voltage", "Current", "Temperature"]]
    train_outputs = train_data[["Capacity"]]
    # Convert the capacity to SoC
    nominal_capacity = 3.0
    train_outputs = train_outputs + nominal_capacity
    soc = train_outputs / nominal_capacity
    train_outputs = soc
    # print(soc)
    test_inputs = test_data[["Step Time", "Voltage", "Current", "Temperature"]]
    test_outputs = test_data[["Capacity"]]
    # Convert the capacity to SoC
    test_outputs = test_outputs + nominal_capacity
    soc = test_outputs / nominal_capacity
    test_outputs = soc
    # Convert the inputs and outputs to tensors
    train_inputs = torch.tensor(train_inputs.values)
    train_outputs = torch.tensor(train_outputs.values)
    test_inputs = torch.tensor(test_inputs.values)
    test_outputs = torch.tensor(test_outputs.values)
    # Convert the inputs and outputs to float
    train_inputs = train_inputs.float()
    train_outputs = train_outputs.float()
    test_inputs = test_inputs.float()
    test_outputs = test_outputs.float()
    # Initiaze the seed for reproducibility
    torch.manual_seed(0)
    # Create the model
    model = PINN_Model()
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # Train the model
    for epoch in range(5000):
        if epoch % 10 == 0:
            # Test the model
            # print(test_inputs.shape)
            outputs = model(test_inputs)
            # Calculate the loss
            loss = model.validation_loss(test_inputs, test_outputs)
            # Print the loss
            print("Test loss: %f" % loss.item())
        # Reset the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(train_inputs)
        # Calculate the loss
        loss = model.loss(train_inputs, train_outputs)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Print the loss
        # print("Epoch: %d, Loss: %f" % (epoch, loss.item()))
    # Test the model
    outputs = model(test_inputs)
    # Calculate the loss
    loss = model.validation_loss(test_inputs, test_outputs)
    # Print the loss
    print("Test loss: %f" % loss.item())
    # Save the model
    torch.save(model.state_dict(), "pth_models/model_soc_pinn.pth")

def cross_validation():
    # Load the dataset
    train_inputs, train_outputs, test_inputs, test_outputs = CustomDataset("panasonic").prepare_panasonic_dataset()
    torch.manual_seed(0)
    # Create the model
    model = PINN_Model()
    # Create the optimizer
    # Define the initial learning rate
    initial_lr = 0.001

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Create a learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)

    # Initialize other training parameters
    best_loss = float("inf")
    # Number of epochs to wait before stopping if validation loss increases
    patience = 20000 

    for epoch in range(500000):
        if epoch % 10 == 0:
            # Calculate the validation loss
            validation_loss = model.validation_loss(test_inputs, test_outputs).item()
            logging.info("Epoch: %d, Validation loss: %f" % (epoch, validation_loss))

            # Update the learning rate scheduler based on the validation loss
            scheduler.step(validation_loss)

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
        loss = model.loss(train_inputs, train_outputs)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
    # Calculate the loss
    loss = model.validation_loss(test_inputs, test_outputs)
    # Print the loss
    print("Test loss: %f" % loss.item())
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
    file_handler = logging.FileHandler("log/train_pinn.log")
    # Set the formatter
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)

if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    # single_dataset()
    # double_dataset()
    cross_validation()
    


