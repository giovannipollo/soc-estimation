import logging
import torch
import pandas as pd
import numpy as np


class SandiaDataset:
    def __init__(
        self,
        file,
        train_cycles=0,
        test_cycles=0,
        physics_cycles=0,
        nominal_capacity=1.1,
        data_top_threshold=1,
        data_bottom_threshold=0,
        physics_top_threshold=1,
        physics_bottom_threshold=0,
        generate_random_data=False,
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
        physics_cycles : int, optional
            Number of cycles to use for physics. The default is 0.
        nominal_capacity : float, optional
            Nominal capacity of the battery. The default is 1.1.
        data_top_threshold : float, optional
            Top threshold for the data. The default is 1.
        data_bottom_threshold : float, optional
            Bottom threshold for the data. The default is 0.
        physics_top_threshold : float, optional
            Top threshold for the physics. The default is 1.
        physics_bottom_threshold : float, optional
            Bottom threshold for the physics. The default is 0.

        Returns
        -------
        None.
        """
        self.train_cycles = train_cycles
        self.test_cycles = test_cycles
        self.physics_cycles = physics_cycles
        self.nominal_capacity = nominal_capacity
        self.data_top_threshold = data_top_threshold
        self.data_bottom_threshold = data_bottom_threshold
        self.physics_top_threshold = physics_top_threshold
        self.physics_bottom_threshold = physics_bottom_threshold
        self.load_dataset(file=file)
        self.reset_time_to_zero_when_new_cycle_starts()
        self.compute_state_of_charge()
        self.clean_dataset()
        self.extract_useful_data()
        self.make_time_relative_previous_timestamp()
        if generate_random_data:
            self.generate_random_data(
                discharge=True, discharge_c=3, charge_c=0.5, num_points=100, temperature=25
            )
        else:
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
        self.data = self.remove_data_below_threshold(
            data=self.data, top_threshold=1, bottom_threshold=0
        )
        # Reset the index
        self.data = self.data.reset_index(drop=True)

    def convert_capacity_to_soc(self):
        """
        Convert the capacity to SoC
        """
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

        # Get the train, test and physics data by filtering the data by cycle
        self.train_data = self.data[self.data["Cycle_Index"].isin(train_cycles)]
        self.test_data = self.data[self.data["Cycle_Index"].isin(test_cycles)]
        self.physics_data = self.data[self.data["Cycle_Index"].isin(physics_cycles)]

        # Remove the data below the threshold
        self.train_data = self.remove_data_below_threshold(
            data=self.train_data,
            top_threshold=self.data_top_threshold,
            bottom_threshold=self.data_bottom_threshold,
        )
        self.physics_data = self.remove_data_below_threshold(
            data=self.physics_data,
            top_threshold=self.physics_top_threshold,
            bottom_threshold=self.physics_bottom_threshold,
        )

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
        if self.train_cycles != 0:
            self.train_data.to_csv("data/temp_data/train_data.csv")
        if self.test_cycles != 0:
            self.test_data.to_csv("data/temp_data/test_data.csv")
        if self.physics_cycles != 0:
            self.physics_data.to_csv("data/temp_data/physics_data.csv")

        # Convert the Test_Time (s) to hours
        # self.train_inputs.loc[:, "Test_Time (s)"] = (
        #     self.train_inputs["Test_Time (s)"] / 3600
        # )
        # self.test_inputs.loc[:, "Test_Time (s)"] = (
        #     self.test_inputs["Test_Time (s)"] / 3600
        # )
        # self.physics_inputs.loc[:, "Test_Time (s)"] = (
        #     self.physics_inputs["Test_Time (s)"] / 3600
        # )

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

    def remove_data_below_threshold(self, data, top_threshold, bottom_threshold):
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
        return data[
            (data["Capacity"] > bottom_threshold) & (data["Capacity"] < top_threshold)
        ]

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
        # Find the indices where Cycle_Index changes
        cycle_indices = np.where(self.data["Cycle_Index"].diff() > 0)[0]

        for idx in cycle_indices:
            initial_time = self.data.at[idx, "Test_Time (s)"]
            self.data.loc[idx:, "Test_Time (s)"] -= initial_time

        # Reset the time for the first cycle
        self.data.loc[self.data.index[0], "Test_Time (s)"] = 0

    def generate_random_data(
        self,
        charge=False,
        discharge=False,
        charge_c=1,
        discharge_c=1,
        num_points=100,
        temperature=25,
    ):
        min_voltage = self.data["Voltage (V)"].min()
        max_voltage = self.data["Voltage (V)"].max()
        charge_time = 1 / charge_c
        discharge_time = 1 / discharge_c
        charge_current = self.nominal_capacity * charge_c
        discharge_current = self.nominal_capacity * discharge_c * -1
        self.physics_data = pd.DataFrame()
        if charge:
            self.physics_data["Test_Time (s)"] = np.linspace(0, charge_time, num_points)
            self.physics_data["Voltage (V)"] = np.linspace(
                min_voltage, max_voltage, num_points
            )
            self.physics_data["Current (A)"] = np.ones(num_points) * charge_current
        else:
            self.physics_data["Test_Time (s)"] = np.linspace(
                charge_time, charge_time + discharge_time, num_points
            )
            self.physics_data["Voltage (V)"] = np.linspace(
                min_voltage, max_voltage, num_points
            )
            self.physics_data["Current (A)"] = np.ones(num_points) * -discharge_current

        self.physics_data["Cell_Temperature (C)"] = np.ones(num_points) * temperature
        self.physics_data.to_csv("data/temp_data/physics_data.csv")
        self.physics_inputs = self.physics_data[
            ["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]
        ]
        self.physics_inputs = torch.tensor(
            self.physics_inputs.values, dtype=torch.float32
        )

    def make_time_relative_previous_timestamp(self):
        """
        Make the time relative to the previous timestamp
        """
        # Get the cycle indexes
        cycle_indexes = self.data["Cycle_Index"].unique()
        # For each cycle
        for cycle_index in cycle_indexes:
            # Get the cycle data
            cycle_data = self.data[self.data["Cycle_Index"] == cycle_index]
            # Get the time steps
            time_steps = cycle_data["Test_Time (s)"]
            # Compute the relative time steps with respect to the previous time step
            relative_time_steps = time_steps.diff()
            # Set the first time step to 0 if the length of the cycle is greater than 1
            if len(relative_time_steps) > 1:
                relative_time_steps.iloc[0] = 0
            # Update the data
            self.data.loc[
                self.data["Cycle_Index"] == cycle_index, "Test_Time (s)"
            ] = relative_time_steps