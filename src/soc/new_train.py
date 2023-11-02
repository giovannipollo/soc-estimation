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
from torch.utils.data import DataLoader


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from dataset.new_sandia import NewSandiaDataset
from plot.plot import Plot


def train():
    # Limit the number of cores used by PyTorch to avoid using all the cores
    torch.set_num_threads(6)

    class_dataset = NewSandiaDataset(
        directory="data/Sandia/time_series_subset",
        cell_type="LFP",
        nominal_capacity=1.1,
        cache=False,
    )

    train_data, test_data, physics_data = class_dataset.split_data_cycles(
        first_part_cycles=1,
        second_part_cycles=1,
        third_part_cycles=1
    )

    train_dataloader = DataLoader(dataset=train_data, batch_size=4096, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(dataset=test_data, batch_size=4096, shuffle=False, num_workers=8)
    physics_dataloader = DataLoader(dataset=physics_data, batch_size=4096, shuffle=False, num_workers=8)


    plot = Plot()


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
    best_train_loss = float("inf")
    best_validation_loss = float("inf")
    # Number of epochs to wait before stopping if validation loss increases
    for batch, (inputs, outputs) in enumerate(train_dataloader):
        train_inputs = inputs
        train_outputs = outputs
        break
    for batch, (inputs, outputs) in enumerate(test_dataloader):
        test_inputs = inputs
        test_outputs = outputs
        break
    for batch, (inputs, outputs) in enumerate(physics_dataloader):
        physics_inputs = inputs
        physics_outputs = outputs
        break
    train_inputs = train_inputs.float()
    train_outputs = train_outputs.float()
    test_inputs = test_inputs.float()
    test_outputs = test_outputs.float()
    physics_inputs = physics_inputs.float()
    physics_outputs = physics_outputs.float()
    # Reshape the outputs to be 2D tensors
    train_outputs = torch.reshape(train_outputs, (-1, 1))
    test_outputs = torch.reshape(test_outputs, (-1, 1))
    physics_outputs = torch.reshape(physics_outputs, (-1, 1))
    logging.debug("Starting training")
    # Set the patience to a huge value to avoid early stopping
    patience = 1000000
    for epoch in range(1500000):

        # Reset the gradients
        optimizer.zero_grad()
        # Calculate the train loss
        train_loss = model.loss(
            x=train_inputs,
            y=train_outputs,
            physics_informed=False,
            physics_x=physics_inputs,
            nominal_capacity=1.1,
        )
        # Calculate the validation loss
        validation_loss = model.validation_loss(x=test_inputs, y=test_outputs).item()
        if epoch % 2000 == 0:
            plot.plot_epoch_predictions_train(
                epoch=epoch,
                model=model,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
            )
            plot.plot_epoch_prediction_test(
                epoch=epoch,
                model=model,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
                validation_loss=validation_loss,
            )
            plot.plot_epoch_prediction_physic(
                epoch=epoch,
                model=model,
                physics_inputs=physics_inputs,
                physics_outputs=physics_outputs,
            )
            plot.plot_epoch_dsoc_dt(
                epoch=epoch,
                model=model,
                physics_inputs=physics_inputs,
                physics_outputs=physics_outputs,
                capacity=1.1,
            )
            logging.info(
                "Epoch: %d, Validation loss: %f, Best Validation loss: %f"
                % (epoch, validation_loss, best_validation_loss)
            )
            # Update the learning rate scheduler based on the validation loss

        # Log the loss
        logging.info("Train loss: %f" % train_loss.item())
        logging.info("Validation loss: %f" % validation_loss)
        # scheduler.step(validation_loss)
        # Check if validation loss is increasing
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            logging.debug("Best validation loss updated: %f" % best_validation_loss)
        else:
            # If validation loss increases for 'patience' epochs, stop training
            if epoch > patience and validation_loss >= best_validation_loss:
                logging.info("Best validation loss: %f" % best_validation_loss)
                logging.info("Early stopping at epoch %d" % epoch)
                break
        if train_loss.item() < best_train_loss:
            best_train_loss = train_loss.item()
            logging.debug("Best train loss: %f" % best_train_loss)
        # Backward pass
        train_loss.backward()
        # Update the weights
        optimizer.step()
    # Calculate the loss
    validation_loss = model.validation_loss(test_inputs, test_outputs)
    # Log the loss
    logging.info("Validation loss: %f" % validation_loss.item())
    logging.info("Best validation loss: %f" % best_validation_loss)
    # Create the gifs
    plot.save_gif_PIL(
        outfile="plots/gif/epoch_predictions_train.gif",
        files=plot.plot_epoch_predictions_train_plots,
    )
    plot.save_gif_PIL(
        outfile="plots/gif/epoch_predictions_test.gif",
        files=plot.plot_epoch_prediction_test_plots,
    )
    plot.save_gif_PIL(
        outfile="plots/gif/epoch_predictions_physics.gif",
        files=plot.plot_epoch_prediction_physic_plots,
    )
    # Save the model
    torch.save(model.state_dict(), "pth_models/model_soc_pinn.pth")


def setup_logging(level=logging.INFO, to_console=False):
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
    # Add the console too for the logger
    if to_console:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)



if __name__ == "__main__":
    # Setup the logging environment withouth debug and info messages
    setup_logging(level=logging.INFO, to_console=False)
    train()
