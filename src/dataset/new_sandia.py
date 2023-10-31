import os
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import torch


class NewSandiaDataset(Dataset):
    def __init__(self, directory, cell_type, nominal_capacity=1.1, cache=False):
        self.directory = directory
        self.cell_type = cell_type
        self.nominal_capacity = nominal_capacity
        if cache:
            self.data = pd.read_pickle("data.pkl")
        else:    
            self.data = self.load_data()
            self.data = self.extract_wanted_data()
            self.data = self.convert_capacity_to_soc()
            self.data = self.remove_data_outside_soc_threshold(top_threshold=1, bottom_threshold=0)
            self.data = self.set_initial_test_time()
            # self.data = self.remove_data_outside_timestamp_threshold(top_threshold=121, bottom_threshold=119)
            self.data.to_csv("data2.csv")
            self.data.to_pickle("data.pkl")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        inputs = data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
        outputs = data["Capacity"]
        return inputs, outputs


    def load_data(self):
        data = []
        absolute_cycle_index = 0
        for file in os.listdir(self.directory):
            if file.endswith(".csv") and file.startswith("SNL_18650_" + self.cell_type):
                df = pd.read_csv(os.path.join(self.directory, file))
                # Add a new column with the absolute cycle index
                df["Absolute_Cycle_Index"] = df["Cycle_Index"] + absolute_cycle_index
                absolute_cycle_index = df["Absolute_Cycle_Index"].iloc[-1]
                df = self.compute_relative_time(df)
                data.append(df)
        return pd.concat(data)

    def extract_wanted_data(self):
        """
        Extract the wanted data from the dataset.
        """
        return self.data[
            [
                "Test_Time (s)",
                "Cycle_Index",
                "Voltage (V)",
                "Current (A)",
                "Cell_Temperature (C)",
                "Charge_Capacity (Ah)",
                "Discharge_Capacity (Ah)",
                "Absolute_Cycle_Index",
            ]
        ]

    def convert_capacity_to_soc(self):
        self.data["Capacity"] = (
            self.data["Charge_Capacity (Ah)"] - self.data["Discharge_Capacity (Ah)"]
        )

        # Convert the capacity to SoC
        self.data["Capacity"] = self.data["Capacity"] / self.nominal_capacity

        # Remove the Charge_Capacity and Discharge_Capacity columns
        self.data = self.data.drop(
            columns=["Charge_Capacity (Ah)", "Discharge_Capacity (Ah)"]
        )
        return self.data

    def remove_data_outside_soc_threshold(self, top_threshold, bottom_threshold):
        """
        Remove the data below the top_threshold and above the bottom_threshold.
        """
        self.data = self.data[
            (self.data["Capacity"] < top_threshold)
            & (self.data["Capacity"] > bottom_threshold)
        ]
        # Reset the index
        self.data = self.data.reset_index(drop=True)
        return self.data

    def remove_data_outside_timestamp_threshold(self, top_threshold, bottom_threshold):
        """
        Remove the data above the threshold.
        """
        self.data = self.data[
            (self.data["Test_Time (s)"] < top_threshold)
            & (self.data["Test_Time (s)"] > bottom_threshold)
        ]
        return self.data

    def split_data(
        self, first_part_percentage, second_part_percentage, third_part_percentage
    ):
        """
        Split the data into train, test and physics data.

        Parameters
        ----------
        train_percentage : float
            Percentage of data used for training.
        test_percentage : float
            Percentage of data used for testing.
        physics_percentage : float
            Percentage of data used for physics.

        Returns
        -------

        """
        # Compute the number of cycles based on the percentage
        train_cycles = int((self.data["Absolute_Cycle_Index"].max()) * first_part_percentage)
        test_cycles = int((self.data["Absolute_Cycle_Index"].max()) * second_part_percentage)
        physics_cycles = int((self.data["Absolute_Cycle_Index"].max()) * third_part_percentage)

        # Get the train data
        first_data = self.data[self.data["Absolute_Cycle_Index"] <= train_cycles]
        
        # Get the test data
        second_data = self.data[
            (self.data["Absolute_Cycle_Index"] > train_cycles)
            & (self.data["Absolute_Cycle_Index"] <= train_cycles + test_cycles)
        ]
        
        # Get the physics data
        third_data = self.data[
            (self.data["Absolute_Cycle_Index"] > train_cycles + test_cycles)
            & (self.data["Absolute_Cycle_Index"] <= train_cycles + test_cycles + physics_cycles)
        ]

        # Create NewSandiaDatasetWrapper objects
        first_dataset = NewSandiaDatasetWrapper(first_data)
        second_dataset = NewSandiaDatasetWrapper(second_data)
        third_dataset = NewSandiaDatasetWrapper(third_data)

        return first_dataset, second_dataset, third_dataset

    def split_data_cycles(self, first_part_cycles, second_part_cycles, third_part_cycles):
        """
        Split the data into train, test and physics data.

        Parameters
        ----------
        train_percentage : float
            Percentage of data used for training.
        test_percentage : float
            Percentage of data used for testing.
        physics_percentage : float
            Percentage of data used for physics.

        Returns
        -------

        """
        # Compute the number of cycles based on the percentage
        train_cycles = first_part_cycles
        test_cycles = second_part_cycles
        physics_cycles = third_part_cycles

        # Get the train data
        first_data = self.data[self.data["Absolute_Cycle_Index"] <= train_cycles]
        
        # Get the test data
        second_data = self.data[
            (self.data["Absolute_Cycle_Index"] > train_cycles)
            & (self.data["Absolute_Cycle_Index"] <= train_cycles + test_cycles)
        ]
        
        # Get the physics data
        third_data = self.data[
            (self.data["Absolute_Cycle_Index"] > train_cycles + test_cycles)
            & (self.data["Absolute_Cycle_Index"] <= train_cycles + test_cycles + physics_cycles)
        ]

        first_data.to_csv("data3.csv")

        # Create NewSandiaDatasetWrapper objects
        first_dataset = NewSandiaDatasetWrapper(first_data)
        second_dataset = NewSandiaDatasetWrapper(second_data)
        third_dataset = NewSandiaDatasetWrapper(third_data)

        return first_dataset, second_dataset, third_dataset


    
    def compute_relative_time(self, data):
        """
        Compute the relative time step with respect to the previous time step for every cycle
        """
        # Get the cycle indexes
        cycle_indexes = data["Cycle_Index"].unique()
        # For each cycle
        for cycle_index in cycle_indexes:
            # Get the cycle data
            cycle_data = data[data["Cycle_Index"] == cycle_index]
            # Get the time steps
            time_steps = cycle_data["Test_Time (s)"]
            # Compute the relative time steps with respect to the previous time step
            relative_time_steps = time_steps.diff()
            # Set the first time step to 0 if the length of the cycle is greater than 1
            if len(relative_time_steps) > 1:
                relative_time_steps.iloc[0] = 0
            # Update the data
            data.loc[
                data["Cycle_Index"] == cycle_index, "Test_Time (s)"
            ] = relative_time_steps
        return data

    def set_initial_test_time(self):
        # Set the first time step to 0
        self.data.loc[0, "Test_Time (s)"] = 0
        return self.data

class NewSandiaDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        inputs = data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
        outputs = data["Capacity"]
        # Convert the inputs and outputs to tensors
        inputs = torch.tensor(inputs)
        outputs = torch.tensor(outputs)
        return inputs, outputs