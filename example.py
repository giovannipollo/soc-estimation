import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def physics_loss_soc_de(model, x, capacity, current = None):
    time_step = x[:, 0].clone().detach().requires_grad_(True)
    voltage = x[:, 1].clone().detach().requires_grad_(True)
    current_in = x[:, 2].clone().detach().requires_grad_(True)
    temperature = x[:, 3].clone().detach().requires_grad_(True)
    # Define the physics inputs
    physics_input = torch.stack((time_step, voltage, current_in, temperature), dim=1)
    # Compute the estimated SoC
    estimated_soc = model.forward(physics_input)
    estimated_soc = torch.flatten(estimated_soc)
    # Compute the derivative of the SoC with respect to time_step
    d_soc_dt = torch.autograd.grad(estimated_soc, time_step, grad_outputs=torch.ones_like(time_step), create_graph=True)[0]
    # Compute the equation loss
    physics_loss_function = nn.MSELoss()
    eq_loss = physics_loss_function(d_soc_dt, current_in / capacity)
    return eq_loss

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)


if __name__ == "__main__":
    # limit the number of threads
    torch.set_num_threads(6)
    # Open the data file
    train_data = pd.read_csv("data/temp_data/train_data.csv")
    # Only keep the value whose Capacity is between 1 and 0.5
    # train_data = train_data[(train_data["Capacity"] < 1) & (train_data["Capacity"] > 0.5)]
    test_data = pd.read_csv("data/temp_data/test_data.csv")
    physics_data = pd.read_csv("data/temp_data/physics_data.csv")
    # Keep only the column with specific names
    train_data = train_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)", "Capacity"]]
    train_input = train_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
    # train_input = train_data[["Test_Time (s)", "Current (A)"]]
    train_output = train_data[["Capacity"]]

    test_data = test_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)", "Capacity"]]
    test_input = test_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
    # test_input = test_data[["Test_Time (s)", "Current (A)"]]
    test_output = test_data[["Capacity"]]

    physics_data = physics_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)", "Capacity"]]
    physics_input = physics_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
    # physics_input = physics_data[["Test_Time (s)", "Current (A)"]]
    physics_current = physics_data[["Current (A)"]]
    physics_output = physics_data[["Capacity"]]
    # Divide the Test_Time by 3600 to get the time in hours
    train_input["Test_Time (s)"] = train_input["Test_Time (s)"] / 3600
    test_input["Test_Time (s)"] = test_input["Test_Time (s)"] / 3600
    physics_input["Test_Time (s)"] = physics_input["Test_Time (s)"] / 3600

    # Convert the data to torch tensors
    train_input = torch.tensor(train_input.values, dtype=torch.float32)
    train_output = torch.tensor(train_output.values, dtype=torch.float32)
    test_input = torch.tensor(test_input.values, dtype=torch.float32)
    test_output = torch.tensor(test_output.values, dtype=torch.float32)
    physics_input = torch.tensor(physics_input.values, dtype=torch.float32)
    physics_output = torch.tensor(physics_output.values, dtype=torch.float32)
    physics_current = torch.tensor(physics_current.values, dtype=torch.float32)
    # Set the torch seed
    torch.manual_seed(0)
    # Define the model of the network to approximate a sine function
    model = nn.Sequential(
        nn.Linear(4, 20), 
        nn.Tanh(), 
        nn.Linear(20, 20), 
        nn.Tanh(), 
        nn.Linear(20, 20), 
        nn.Tanh(), 
        nn.Linear(20, 20),
        nn.Tanh(),
        nn.Linear(20, 1)
        )
    # Define the loss function
    loss_function = nn.MSELoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Define the number of epochs to train for
    num_epochs = 50000
    # Define the number of training examples
    num_train_examples = 100
    # Define the number of test examples
    num_test_examples = 100
    # Initialize other training parameters
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    # Define the input data 
    # input_data = torch.linspace(0, 2 * 3.1415, num_train_examples).reshape(-1, 1)
    # output_data = torch.sin(input_data)
    # test_input_data = torch.linspace(0, 2 * 3.1415, num_test_examples).reshape(-1, 1)
    # test_output_data = torch.sin(test_input_data)
    # Train the model
    figures = []
    for epoch in range(num_epochs):
        # Compute the validation loss
        output = model(test_input)
        validation_loss = loss_function(output, test_output)
        print("Epoch: %d, Validation loss: %f" % (epoch, validation_loss.item()))
        # Zero the gradients
        optimizer.zero_grad()
        # Compute the output of the model
        output = model(train_input)
        # Compute the loss
        loss = loss_function(output, train_output)
        # Add the physics loss
        physics_loss = 0
        # if epoch < 10000:
        #     physics_loss = 0
        # elif epoch < 30000:
        #     lr = 0.000001
        #     loss = physics_loss_soc_de(model=model, x=physics_input, capacity=1.1, current=physics_current)
        # else:
        #     loss = 0.01*loss + 0.1*physics_loss_soc_de(model=model, x=physics_input, capacity=1.1, current=physics_current)

        loss = loss + 0.0001*physics_loss_soc_de(model=model, x=physics_input, capacity=1.1, current=physics_current)
        # Compute the gradients
        # loss = loss + physics_loss
        loss.backward()
        # Update the weights
        optimizer.step()
        # Print the loss
        print("Epoch: %d, Train loss: %f" % (epoch, loss.item()))
        print("Epoch: %d, Physics loss: %f" % (epoch, physics_loss))
        # Check if the validation loss is the best
        if validation_loss.item() < best_val_loss:
            best_val_loss = validation_loss.item()
            print("Best validation loss: %f" % best_val_loss)
            # Save the model weights
            torch.save(model.state_dict(), "model_weights.pth")
            output = model(test_input)
            plt.figure()
            # Plot the ground truth with the test time on the x-axis
            plt.scatter(test_input[:, 0].detach().numpy(), output.detach().numpy(), label="Prediction")
            plt.scatter(test_input[:, 0].detach().numpy(), test_output.detach().numpy(), label="Ground truth")
            plt.legend()
            plt.savefig("plots/test_prediction_epoch_best.png")
            plt.close()
        # if loss.item() < best_train_loss:
        #     best_train_loss = loss.item()
        #     output = model(train_input)
        #     plt.figure()
        #     # Plot the ground truth with the test time on the x-axis
        #     plt.scatter(train_input[:, 0].detach().numpy(), output.detach().numpy(), label="Prediction")
        #     plt.scatter(train_input[:, 0].detach().numpy(), train_output.detach().numpy(), label="Ground truth")
        #     plt.legend()
        #     plt.savefig("plots/train_prediction_epoch_best.png")
        #     plt.close()
        # Plot the prediction for the test data
        if epoch % 300 == 0:
            output = model(test_input)
            plt.figure(figsize=(8,4))
            l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
            plt.setp(l.get_texts(), color="k")
            # Place the epoch number on the plot
            plt.text(0.5, 1.1, "Epoch: %d" % epoch, transform=plt.gca().transAxes)
            # Plot the ground truth with the test time on the x-axis
            plt.plot(test_input[:, 0].detach().numpy(), output.detach().numpy(), label="Prediction")
            plt.plot(test_input[:, 0].detach().numpy(), test_output.detach().numpy(), label="Ground truth")
            plt.legend()
            filename = "plots/test_prediction_epoch_%i.png"%(epoch)
            plt.savefig(filename)
            figures.append(filename)
            plt.close()
        # Plot the prediction for the train data
        if epoch % 300 == 0:
            output = model(train_input)
            plt.figure()
            # Plot the ground truth with the test time on the x-axis
            plt.plot(train_input[:, 0].detach().numpy(), output.detach().numpy(), label="Prediction")
            plt.plot(train_input[:, 0].detach().numpy(), train_output.detach().numpy(), label="Ground truth")
            plt.legend()
            plt.savefig("plots/train_prediction_epoch.png")
            plt.close()

    save_gif_PIL("nn.gif", figures, fps=20, loop=0)