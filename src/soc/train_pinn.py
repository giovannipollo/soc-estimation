import torch.nn as nn
import pandas as pd
import torch
from model import Model
from pinn_model import PINN_Model

if __name__ == "__main__":
    # Limit the number of cpu cores to 1
    # torch.set_num_threads(16)
    # Load the csv file with the data
    data = pd.read_csv("data/549_HPPC.csv")
    # Extract the data we want to use
    data = data[["Voltage", "Current", "Temperature", "Capacity"]]
    # Split the data into train and test
    train_data = data.sample(frac=0.2, random_state=0)
    test_data = data.drop(train_data.index)
    # Extract the inputs and outputs
    train_inputs = train_data[["Voltage", "Current", "Temperature"]]
    train_outputs = train_data[["Capacity"]]
    # Convert the capacity to SoC
    nominal_capacity = 3.0
    train_outputs = train_outputs + nominal_capacity
    soc = train_outputs / nominal_capacity
    train_outputs = soc
    test_inputs = test_data[["Voltage", "Current", "Temperature"]]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Train the model
    for epoch in range(20000):
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
