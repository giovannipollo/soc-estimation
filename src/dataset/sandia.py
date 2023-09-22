import logging
import torch    
import pandas as pd

class SandiaDataset():
    
    def __init__(self, file, train_split=0.8, validation_split=0.2, nominal_capacity=1.1):
        """
        Constructor of the dataset.

        Parameters
        ----------
        file : str
            Path to the dataset file.
        train_split : float, optional
            Percentage of the dataset to use for training. The default is 0.8.
        validation_split : float, optional
            Percentage of the dataset to use for validation. The default is 0.2.
        nominal_capacity : float, optional
            Nominal capacity of the battery. The default is 1.1.

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
        self.train_split = train_split
        self.validation_split = validation_split
        self.nominal_capacity = nominal_capacity
        self.load_dataset(file=file)
        self.clean_dataset()
        self.compute_state_of_charge()
        self.extract_useful_data()
        self.split_and_prepare_dataset()
        return self.train_inputs, self.train_outputs, self.test_inputs, self.test_outputs

    def load_dataset(self, file):
        """Prepare sandia dataset"""
        self.data = pd.read_csv(file)
        # Extract the data we want to use
        self.data = self.data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)", "Charge_Capacity (Ah)", "Discharge_Capacity (Ah)"]]

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
        self.data["Capacity"] = self.data["Charge_Capacity (Ah)"] - self.data["Discharge_Capacity (Ah)"]
        self.data["Capacity"] = self.data["Capacity"] / self.nominal_capacity
    
    def compute_state_of_charge(self):
        """
        Compute the actual capacity
        """
        self.data["Capacity"] = self.data["Charge_Capacity (Ah)"] - self.data["Discharge_Capacity (Ah)"]
        # Convert the capacity to SoC
        self.convert_capacity_to_soc()

    def extract_useful_data(self):
        """
        Extract the useful data from the dataset
        """
        self.data = self.data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)", "Capacity"]]
    
    def split_and_prepare_dataset(self):
        self.train_data = self.data.sample(frac=self.train_split, random_state=1)
        self.test_data = self.data.drop(self.train_data.index)
        self.train_inputs = self.train_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
        self.train_outputs = self.train_data[["Capacity"]]
        self.test_inputs = self.test_data[["Test_Time (s)", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
        self.test_outputs = self.test_data[["Capacity"]]
        # Convert the inputs and outputs to tensors
        self.train_inputs = torch.tensor(self.train_inputs.values)
        self.train_outputs = torch.tensor(self.train_outputs.values)
        self.test_inputs = torch.tensor(self.test_inputs.values)
        self.test_outputs = torch.tensor(self.test_outputs.values)
        # Convert the inputs and outputs to float
        self.train_inputs = self.train_inputs.float()
        self.train_outputs = self.train_outputs.float()
        self.test_inputs = self.test_inputs.float()
        self.test_outputs = self.test_outputs.float()