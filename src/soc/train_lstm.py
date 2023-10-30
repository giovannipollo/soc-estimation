import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torch
from lstm_model import LSTM_Model
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
    data = NewSandiaDataset(
        directory="data/Sandia/time_series",
        cell_type="LFP",
        nominal_capacity=1.1,
        cache=True
    )

    plot = Plot()
    train_data, physics_data, test_data = data.split_data(
        first_part_percentage=0.8,
        second_part_percentage=0.1,
        third_part_percentage=0.1,
    )

    print("arrivato")
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    physics_dataloader = DataLoader(physics_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    for i, (inputs, outputs) in enumerate(train_dataloader):
        print(inputs)
        print(outputs)
        if i == 10:
            break

    print("arrivato")

    # Create the model
    model = LSTM_Model(input_size=4, output_size=1, hidden_size=16)

    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create the scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)

    # Initialize other training parameters
    best_train_loss = float("inf")
    best_validation_loss = float("inf")

    for epoch in range(150000):
        # Reset the gradients
        optimizer.zero_grad()
        # Forward pass
        pred_outputs = model(train_inputs)
        # Calculate the train loss
        train_loss = model.loss(
            y_pred=pred_outputs,
            y_true=train_outputs,
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
        scheduler.step(train_loss)
        print("Scheduler: ", scheduler.state_dict())
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
