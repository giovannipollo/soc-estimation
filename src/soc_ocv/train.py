import torch.nn as nn
import pandas as pd
import torch
from model_soc_ocv import Model_Soc_Ocv

if __name__ == "__main__":
    # Limit the number of cpu cores to 1
    # torch.set_num_threads(1)
    # Load the csv file with the data
    data = pd.read_csv("data/549_HPPC.csv")
    data = data.loc[data["Status"] == "PAU"]
    # Remove data that are equal between them
    data = data.drop_duplicates(subset=["Temperature", "Capacity"])
    # Extract the data we want to use
    data = data[["Voltage", "Temperature", "Capacity"]]
    # Split the data into train and test
    train_data = data.sample(frac=0.8, random_state=0)
    test_data = data.drop(train_data.index)
    # Extract the inputs and outputs
    train_inputs = train_data[["Temperature", "Capacity"]]
    train_outputs = train_data[["Voltage"]]
    # Convert the capacity to SoC
    nominal_capacity = 3.0
    train_inputs["Capacity"] = train_inputs["Capacity"] + nominal_capacity
    soc = train_inputs["Capacity"] / nominal_capacity
    train_inputs["Capacity"] = soc

    test_inputs = test_data[["Temperature", "Capacity"]]
    test_outputs = test_data[["Voltage"]]
    # Convert the capacity to SoC
    test_inputs["Capacity"] = test_inputs["Capacity"] + nominal_capacity
    soc = test_inputs["Capacity"] / nominal_capacity
    test_inputs["Capacity"] = soc
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
    # Create the model
    model = Model_Soc_Ocv()
    # model = PINN_Model()
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Train the model
    for epoch in range(1000):
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
        print("Epoch: %d, Loss: %f" % (epoch, loss.item()))
        if epoch % 10 == 0:
            # Test the model
            outputs = model(test_inputs)
            # Calculate the loss
            loss = model.loss(test_inputs, test_outputs)
            # Print the loss
            print("Test loss: %f" % loss.item())
    # Test the model
    outputs = model(test_inputs)
    # Calculate the loss
    loss = model.loss(test_inputs, test_outputs)
    # Print the loss
    print("Test loss: %f" % loss.item())
    # Save the model
    torch.save(model.state_dict(), "pth_models/model_soc_ocv.pth")
    # Plot the results of the test
    import matplotlib.pyplot as plt
    # Plot the Voltage vs Capacity
    plt.figure()
    plt.plot(test_inputs[:, 1], test_outputs, label="True")
    plt.plot(test_inputs[:, 1], outputs.detach().numpy(), label="Predicted")
    plt.xlabel("Capacity")
    plt.ylabel("Voltage")
    plt.legend()
    plt.savefig("results.png")

