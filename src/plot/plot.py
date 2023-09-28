import matplotlib.pyplot as plt
from PIL import Image


class Plot:
    def __init__(self):
        self.plot_epoch_predictions_train_plots = []
        self.plot_epoch_prediction_test_plots = []
        self.plot_epoch_prediction_physic_plots = []

    def get_plot_epoch_predictions_train_plots(self):
        return self.plot_epoch_predictions_train_plots
    
    def get_plot_epoch_prediction_test_plots(self):
        return self.plot_epoch_prediction_test_plots
    
    def get_plot_epoch_prediction_physic_plots(self):
        return self.plot_epoch_prediction_physic_plots
    
    def plot_epoch_prediction_test(self, epoch, model, test_inputs, test_outputs, validation_loss):
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
        plt.figure(figsize=(8,4))
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        # Place the epoch number on the plot
        plt.text(0.3, 1.1, "Epoch: %d, MSELoss: %f" % (epoch, validation_loss), transform=plt.gca().transAxes)
        plt.scatter(
            test_inputs[:, 0].detach().numpy(),
            test_outputs.detach().numpy(),
            label="True SoC",
        )
        plt.scatter(
            test_inputs[:, 0].detach().numpy(),
            model.forward(test_inputs).detach().numpy(),
            label="Predicted SoC",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("SoC")
        plt.legend()
        filename = "plots/test/epoch_%d_test.png" % epoch
        plt.savefig(filename)
        self.plot_epoch_prediction_test_plots.append(filename)
        plt.close()

    def plot_epoch_predictions_train(self, epoch, model, train_inputs, train_outputs):
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
        plt.figure(figsize=(8,4))
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        # Place the epoch number on the plot
        plt.text(0.5, 1.1, "Epoch: %d" % epoch, transform=plt.gca().transAxes)
        # Plot the points without the line
        plt.scatter(
            train_inputs[:, 0].detach().numpy(),
            train_outputs.detach().numpy(),
            label="True SoC",
            s=1,
        )
        plt.scatter(
            train_inputs[:, 0].detach().numpy(),
            model.forward(train_inputs).detach().numpy(),
            label="Predicted SoC",
            s=1,
        )
        plt.xlabel("Time (s)")
        plt.ylabel("SoC")
        plt.legend()
        filename = "plots/train/epoch_%d_train.png" % epoch
        plt.savefig(filename)
        self.plot_epoch_predictions_train_plots.append(filename)
        plt.close()

    def plot_epoch_prediction_physic(
        self, epoch, model, physics_inputs, physics_outputs
    ):
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
        plt.figure(figsize=(8,4))
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        # Place the epoch number on the plot
        plt.text(0.5, 1.1, "Epoch: %d" % epoch, transform=plt.gca().transAxes)
        plt.scatter(
            physics_inputs[:, 0].detach().numpy(),
            model.forward(physics_inputs).detach().numpy(),
            label="Predicted SoC Physics",
            s=1,
        )
        plt.scatter(
            physics_inputs[:, 0].detach().numpy(),
            physics_outputs.detach().numpy(),
            label="True SoC Physics",
            s=1,
        )
        plt.scatter
        plt.xlabel("Time (s)")
        plt.ylabel("SoC")
        plt.legend()
        filename = "plots/physics/epoch_%d_physics.png" % epoch
        plt.savefig(filename)
        self.plot_epoch_prediction_physic_plots.append(filename)
        plt.close()

    def save_gif_PIL(self, outfile, files, fps=5, loop=0):
        imgs = [Image.open(file) for file in files]
        imgs[0].save(
            fp=outfile,
            format="GIF",
            append_images=imgs[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=loop,
        )
