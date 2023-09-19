class CustomDataset():
    def __init__(self, dataset_type):
        if dataset_type == "panasonic":
            self.prepare_panasonic_dataset()
        elif dataset_type == "lg":
            self.prepare_lg_dataset()
        else:
            raise ValueError("Invalid dataset type")

    def prepare_panasonic_dataset():
        """Prepare panasonic dataset"""
        pass

    def prepare_lg_dataset():
        """Prepare lg dataset"""
        pass
