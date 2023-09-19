import pandas as pd
import torch
import scipy.io as sio

class CustomDataset():
    def __init__(self):
        pass
    
    def prepare_panasonic_dataset(self, file="data/Panasonic/Panasonic 18650PF Data/Panasonic 18650PF Data/25degC/Drive cycles/03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat", train_split=0.8, validation_split=0.2):
        """Prepare panasonic dataset"""
        # Load the .mat file with the test data
        data = sio.loadmat(file)
        # Extract the data we want to use
        data = data["meas"]
        # Extract the data we want to use
        timestamp = torch.Tensor(data["Time"][0][0]).flatten()
        voltage = torch.Tensor(data["Voltage"][0][0]).flatten()
        current = torch.Tensor(data["Current"][0][0]).flatten()
        temperature = torch.Tensor(data["Battery_Temp_degC"][0][0]).flatten()
        capacity = torch.Tensor(data["Ah"][0][0]).flatten()
        # Define the new data
        data = {
            "Step Time": timestamp,
            "Voltage": voltage,
            "Current": current,
            "Temperature": temperature,
            "Capacity": capacity
        }
        # Convert the data to a pandas dataframe
        data = pd.DataFrame(data)
        # Split the data into train and test
        train_data = data.sample(frac=train_split, random_state=1)
        test_data = data.drop(train_data.index)
        # Extract the inputs and outputs
        train_inputs = train_data[["Step Time", "Voltage", "Current", "Temperature"]]
        train_outputs = train_data[["Capacity"]]
        # Convert the capacity to SoC
        self.convert_capacity_to_soc(train_outputs)
        test_inputs = test_data[["Step Time", "Voltage", "Current", "Temperature"]]
        test_outputs = test_data[["Capacity"]]
        # Convert the capacity to SoC
        self.convert_capacity_to_soc(test_outputs)
        # Convert the inputs and outputs to tensors
        train_inputs = torch.tensor(train_inputs.values)
        train_outputs = torch.tensor(train_outputs.values)
        test_inputs = torch.tensor(test_inputs.values)
        test_outputs = torch.tensor(test_outputs.values)
        # Return the data
        return train_inputs, train_outputs, test_inputs, test_outputs
    
    def prepare_step_time(self, data):
        """
        Convert the step time from hh:mm:ss.ms to ss.ms.
        """
        for i in range(len(data["Step Time"])):
            step_time = data["Step Time"][i]
            step_time = step_time.split(":")
            step_time = float(step_time[0]) * 3600 + float(step_time[1]) * 60 + float(step_time[2])
            data["Step Time"][i] = step_time
        data["Step Time"] = data["Step Time"].astype(float)


    def convert_capacity_to_soc(self, data, nominal_capacity = 3.0):
        """Convert the capacity from the format Ah to the format SoC (0-1)"""
        return ((data + nominal_capacity)/nominal_capacity)

    def prepare_lg_dataset(self, file="data/549_HPPC.csv", train_split=0.8, validation_split=0.2):
        """Prepare lg dataset"""
        # Load the csv file with the data
        data = pd.read_csv(file)
        # Extract the data we want to use
        data = data[["Step Time", "Voltage", "Current", "Temperature", "Capacity"]]
        # Prepare the step time
        self.prepare_step_time(data)
        # Split the data into train and test
        train_data = data.sample(frac=train_split, random_state=1)
        test_data = data.drop(train_data.index)
        # Extract the inputs and outputs
        train_inputs = train_data[["Step Time", "Voltage", "Current", "Temperature"]]
        train_outputs = train_data[["Capacity"]]
        # Convert the capacity to SoC
        self.convert_capacity_to_soc(train_outputs)
        test_inputs = test_data[["Step Time", "Voltage", "Current", "Temperature"]]
        test_outputs = test_data[["Capacity"]]
        # Convert the capacity to SoC
        self.convert_capacity_to_soc(test_outputs)
        # Convert the inputs and outputs to tensors
        train_inputs = torch.tensor(train_inputs.values)
        train_outputs = torch.tensor(train_outputs.values)
        test_inputs = torch.tensor(test_inputs.values)
        test_outputs = torch.tensor(test_outputs.values)
        # Convert the inputs and outputs to float
        train_inputs = torch.tensor(train_inputs.values)
        train_outputs = torch.tensor(train_outputs.values)
        test_inputs = torch.tensor(test_inputs.values)
        test_outputs = torch.tensor(test_outputs.values)
        # Return the data
        return train_inputs, train_outputs, test_inputs, test_outputs
    
    def prepare_sandia_time_series(self, file="data/Sandia/time_series/SNL_18650_LFP_15C_0-100_0.5-1C_a_timeseries.csv", train_split=0.8, validation_split=0.2):
        """Prepare sandia dataset"""
        data = pd.read_csv(file)
        # Extract the data we want to use
        data = data[["Cycle_Index", "Voltage (V)", "Current (A)", "Cell_Temperature (C)", "Charge_Capacity (Ah)", "Discharge_Capacity (Ah)"]]
        # Compute the actual capacity
        data["Capacity"] = data["Charge_Capacity (Ah)"] - data["Discharge_Capacity (Ah)"]
        # Convert the capacity to SoC
        self.convert_capacity_to_soc(data["Capacity"], nominal_capacity=1.1)
        data = data[["Cycle_Index", "Voltage (V)", "Current (A)", "Cell_Temperature (C)", "Capacity"]]
        # Split the data into train and test
        train_data = data.sample(frac=train_split, random_state=1)
        test_data = data.drop(train_data.index)
        # Extract the inputs and outputs
        train_inputs = train_data[["Cycle_Index", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
        train_outputs = train_data[["Capacity"]]
        test_inputs = test_data[["Cycle_Index", "Voltage (V)", "Current (A)", "Cell_Temperature (C)"]]
        test_outputs = test_data[["Capacity"]]
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

        # Return the data
        return train_inputs, train_outputs, test_inputs, test_outputs

        
