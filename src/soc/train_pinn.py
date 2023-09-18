import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torch
from model import Model
from pinn_model import PINN_Model
import scipy.io as sio

def prepare_step_time(data):
    """
    Convert the step time from hh:mm:ss.ms to ss.ms.
    """
    for i in range(len(data["Step Time"])):
        step_time = data["Step Time"][i]
        step_time = step_time.split(":")
        step_time = float(step_time[0]) * 3600 + float(step_time[1]) * 60 + float(step_time[2])
        data["Step Time"][i] = step_time
    data["Step Time"] = data["Step Time"].astype(float)

def single_dataset():
    # Limit the number of cpu cores to 1
    # torch.set_num_threads(16)
    # Load the csv file with the data
    data = pd.read_csv("data/549_HPPC.csv")
    # Extract the data we want to use
    data = data[["Step Time", "Voltage", "Current", "Temperature", "Capacity"]]
    # Prepare the step time
    prepare_step_time(data)
    # Split the data into train and test
    train_data = data.sample(frac=0.2, random_state=0)
    test_data = data.drop(train_data.index)
    # Extract the inputs and outputs
    train_inputs = train_data[["Step Time", "Voltage", "Current", "Temperature"]]
    train_outputs = train_data[["Capacity"]]
    # Convert the capacity to SoC
    nominal_capacity = 3.0
    train_outputs = train_outputs + nominal_capacity
    soc = train_outputs / nominal_capacity
    train_outputs = soc
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
    # Limit the number of cpu cores to 1
    # torch.set_num_threads(16)
    # Load the csv file with the data
    data = pd.read_csv("data/549_HPPC.csv")
    # Extract the data we want to use
    data = data[["Step Time", "Voltage", "Current", "Temperature", "Capacity"]]
    # Prepare the step time
    prepare_step_time(data)
    # Load the .mat file with the test data
    test_data = sio.loadmat("data/03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat")
    # Extract the data we want to use
    test_data = test_data["meas"]
    # Split the data in train, test and validation
    train_data = data.sample(frac=0.8, random_state=0)
    physics_data = data.drop(train_data.index)
    # Extract the inputs and outputs
    train_inputs = train_data[["Step Time", "Voltage", "Current", "Temperature"]]
    train_outputs = train_data[["Capacity"]]
    physics_input = physics_data[["Step Time", "Voltage", "Current", "Temperature"]]
    # Convert the capacity to SoC
    nominal_capacity_train = 3.0
    train_outputs = train_outputs + nominal_capacity_train
    soc = train_outputs / nominal_capacity_train
    train_outputs = soc
    # print(soc)
    # Prepare the test inputs and outputs
    timestamp = torch.Tensor(test_data["Time"][0][0]).flatten()
    voltage = torch.Tensor(test_data["Voltage"][0][0]).flatten()
    current = torch.Tensor(test_data["Current"][0][0]).flatten()
    battery_temp_degC = torch.Tensor(test_data["Battery_Temp_degC"][0][0]).flatten()
    test_inputs = torch.stack((timestamp, voltage, current, battery_temp_degC), dim=1)
    test_outputs = test_data[["Ah"]]
    test_outputs = test_outputs[0][0][0]
    # Convert the capacity to SoC
    nominal_capacity_test = 2.9
    test_outputs = torch.Tensor(test_outputs) + nominal_capacity_test
    soc = test_outputs / nominal_capacity_test
    test_outputs = soc
    # Convert the inputs and outputs to tensors
    train_inputs = torch.tensor(train_inputs.values)
    train_outputs = torch.tensor(train_outputs.values)
    physics_input = torch.tensor(physics_input.values)
    # Convert the inputs and outputs to float
    train_inputs = train_inputs.float()
    train_outputs = train_outputs.float()
    test_outputs = test_outputs.float()
    test_inputs = test_inputs.float()
    physics_input = physics_input.float()
    # Initiaze the seed for reproducibility
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
    patience = 20000  # Number of epochs to wait before stopping if validation loss increases

    for epoch in range(500000):
        if epoch % 10 == 0:
            # Calculate the validation loss
            validation_loss = model.validation_loss(test_inputs, test_outputs).item()
            print("Validation loss: %f" % validation_loss)

            # Update the learning rate scheduler based on the validation loss
            scheduler.step(validation_loss)

            # Check if validation loss is increasing
            if validation_loss < best_loss:
                best_loss = validation_loss
                print("Best validation loss: %f" % best_loss)
                # Save the model weights
                torch.save(model.state_dict(), "best_model.pth")
            else:
                # If validation loss increases for 'patience' epochs, stop training
                if epoch > patience and validation_loss >= best_loss:
                    print("Best validation loss: %f" % best_loss)
                    print("Early stopping at epoch %d" % epoch)
                    break
         # Reset the gradients
        optimizer.zero_grad()
        # Calculate the loss
        loss = model.loss(train_inputs, train_outputs, physics_input)
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


if __name__ == "__main__":
    # single_dataset()
    # double_dataset()
    cross_validation()
    


