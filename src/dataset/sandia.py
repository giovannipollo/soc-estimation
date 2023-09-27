import logging
import torch
import pandas as pd
import numpy as np


class SandiaDataset:
    def __init__(
        self,
        file,
        train_cycles=80,
        test_cycles=20,
        physics_cycles=10,
        nominal_capacity=1.1,
        threshold=0,
    ):
        """
        Constructor of the dataset.

        Parameters
        ----------
        file : str
            Path to the dataset file.
        train_cycles : int, optional
            Number of cycles to use for training. The default is 0.
        test_cycles : int, optional
            Number of cycles to use for testing. The default is 0.
        nominal_capacity : float, optional
            Nominal capacity of the battery. The default is 1.1.
        threshold : float, optional
            Threshold for the SoC. The default is 0. This means that the network is trained with data where the SoC is between 1 and 0. Allowed values are between 0 and 1.

        Returns
        -------
        train_inputs : torch.tensor
            Inputs for training.
        train_outputs : torch.tensor
            Outputs for training.
        test_inputs : torch.tensor
            Inputs for testing.
        test_outputs : torch.tensor
            Outputs for testing.
        """
        self.train_cycles = train_cycles
        self.test_cycles = test_cycles
        self.physics_cycles = physics_cycles
        self.nominal_capacity = nominal_capacity
        self.threshold = threshold
        self.load_dataset(file=file)
        self.clean_dataset()
        self.reset_time_to_zero_when_new_cycle_starts()
        self.compute_state_of_charge()
        self.extract_useful_data()
        self.split_and_prepare_dataset()

    def load_dataset(self, file):
        """
        Prepare sandia dataset

        Parameters
        ----------
        file : str
            Path to the dataset file.

        Returns
        -------
        None.
        """
        self.data = pd.read_csv(file)
        # Extract the data we want to use
        self.data = self.data[
            [
                "Test_Time (s)",
                "Cycle_Index",
                "Voltage (V)",
                "Current (A)",
                "Cell_Temperature (C)",
                "Charge_Capacity (Ah)",
                "Discharge_Capacity (Ah)",
            ]
        ]

    def clean_dataset(self):
        """
        Clean the dataset, mainly removing the lines where the difference between the charge and discharge capacity is smaller than zero
        """

        for line in self.data.iterrows():
            if line[1]["Charge_Capacity (Ah)"] - line[1]["Discharge_Capacity (Ah)"] < 0:
                logging.debug("Removing line %d" % line[0])
                self.data.drop(line[0], inplace=True)

    def convert_capacity_to_soc(self):
        """
        Convert the capacity to SoC
        """
        self.data["Capacity"] = (
            self.data["Charge_Capacity (Ah)"] - self.data["Discharge_Capacity (Ah)"]
        )
        self.data["Capacity"] = self.data["Capacity"] / self.nominal_capacity

    def compute_state_of_charge(self):
        """
        Compute the actual capacity and convert it to SoC
        """
        self.data["Capacity"] = (
            self.data["Charge_Capacity (Ah)"] - self.data["Discharge_Capacity (Ah)"]
        )
        # Convert the capacity to SoC
        self.convert_capacity_to_soc()

    def extract_useful_data(self):
        """
        Extract the useful data from the dataset
        """
        self.data = self.data[
            [
                "Test_Time (s)",
                "Cycle_Index",
                "Voltage (V)",
                "Current (A)",
                "Cell_Temperature (C)",
                "Capacity",
            ]
        ]

    def split_and_prepare_dataset(self):
        """
        Split the dataset into train and test by using the specified number of cycles taken randomly from the dataset
        """
        # Get the number of cycles
        # Save the data to a csv file
        cycles = self.data["Cycle_Index"].unique()
        # Shuffle the cycles
        # np.random.shuffle(cycles)
        # Get the train and test cycles
        train_cycles = cycles[0 : self.train_cycles]
        test_cycles = cycles[self.train_cycles : self.train_cycles + self.test_cycles]
        physics_cycles = cycles[
            self.train_cycles
            + self.test_cycles : self.train_cycles
            + self.test_cycles
            + self.physics_cycles
        ]
        # Get the train and test data
        self.train_data = self.data[self.data["Cycle_Index"].isin(train_cycles)]
        self.test_data = self.data[self.data["Cycle_Index"].isin(test_cycles)]
        self.physics_data = self.data[self.data["Cycle_Index"].isin(physics_cycles)]
        # Remove the data below the threshold
        self.train_data = self.remove_data_below_threshold(data=self.train_data)
        # Extract the inputs and outputs
        self.train_inputs = self.train_data[
            ["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]
        ]
        self.train_outputs = self.train_data[["Capacity"]]
        self.test_inputs = self.test_data[
            ["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]
        ]
        self.test_outputs = self.test_data[["Capacity"]]
        self.physics_inputs = self.physics_data[
            ["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]
        ]
        self.physics_outputs = self.physics_data[["Capacity"]]
        # Save the train and test data
        self.train_data.to_csv("data/temp_data/train_data.csv")
        self.test_data.to_csv("data/temp_data/test_data.csv")
        self.physics_data.to_csv("data/temp_data/physics_data.csv")
        # Convert the Test_Time (s) to hours
        self.train_inputs["Test_Time (s)"] = self.train_inputs["Test_Time (s)"] / 3600
        self.test_inputs["Test_Time (s)"] = self.test_inputs["Test_Time (s)"] / 3600
        self.physics_inputs["Test_Time (s)"] = (
            self.physics_inputs["Test_Time (s)"] / 3600
        )
        # Convert the inputs and outputs to tensors
        self.train_inputs = torch.tensor(self.train_inputs.values, dtype=torch.float32)
        self.train_outputs = torch.tensor(
            self.train_outputs.values, dtype=torch.float32
        )
        self.test_inputs = torch.tensor(self.test_inputs.values, dtype=torch.float32)
        self.test_outputs = torch.tensor(self.test_outputs.values, dtype=torch.float32)
        self.physics_inputs = torch.tensor(
            self.physics_inputs.values, dtype=torch.float32
        )
        self.physics_outputs = torch.tensor(
            self.physics_outputs.values, dtype=torch.float32
        )

    def remove_data_below_threshold(self, data):
        """
        Remove the data below the specified threshold

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe to remove the data from.

        Returns
        -------
        None.
        """
        return data[data["Capacity"] > self.threshold]

    def get_train_data(self):
        """
        Get the train data

        Returns
        -------
        train_inputs : torch.tensor
            Inputs for training.
        train_outputs : torch.tensor
            Outputs for training.
        """
        return self.train_inputs, self.train_outputs

    def get_test_data(self):
        """
        Get the test data

        Returns
        -------
        test_inputs : torch.tensor
            Inputs for testing.
        test_outputs : torch.tensor
            Outputs for testing.
        """
        return self.test_inputs, self.test_outputs

    def get_physics_input(self):
        """
        Get the physics data

        Returns
        -------
        physics_inputs : torch.tensor
            Inputs for physics.
        physics_outputs : torch.tensor
            Outputs for physics.
        """
        return self.physics_inputs

    def get_physics_output(self):
        """
        Get the physics data

        Returns
        -------
        physics_inputs : torch.tensor
            Inputs for physics.
        physics_outputs : torch.tensor
            Outputs for physics.
        """
        return self.physics_outputs

    def reset_time_to_zero_when_new_cycle_starts(self):
        """
        Reset the time to zero when a new cycle starts
        """
        current_cycle_index = 1.0
        initial_time = self.data.iloc[0]["Test_Time (s)"]
        for line in self.data.iterrows():
            if line[1]["Cycle_Index"] > current_cycle_index:
                current_cycle_index = line[1]["Cycle_Index"]
                initial_time = line[1]["Test_Time (s)"]
                logging.debug("Resetting time to zero for line %d" % line[0])
                self.data.at[line[0], "Test_Time (s)"] = 0
            else:
                self.data.at[line[0], "Test_Time (s)"] = (
                    line[1]["Test_Time (s)"] - initial_time
                )
