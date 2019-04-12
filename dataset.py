import pickle
from pathlib import Path

import numpy as np
import torch.utils.data as data

from baseline import create_instance_dataset


class InstanceDataset(data.Dataset):
    def __init__(self):
        preprocessed_dataset_file = Path("preprocessed_data.pkl")
        file_stream = open(preprocessed_dataset_file, 'rb')
        df = pickle.load(file_stream)

        self.frame = create_instance_dataset(df)

        self.frame = self.filter_nan(self.frame)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, item):
        targets = self.frame[item][3]
        df = self.frame[item][2]
        inputs = df.values.ravel()

        return inputs.astype(np.float32), targets.astype(np.float32)

    def filter_nan(self, df):
        def thefilter(x):
            rv = not x[2].isnull().values.any() and not np.isnan(x[3])
            return rv

        return list(filter(thefilter, df))


if __name__ == '__main__':
    dataset = InstanceDataset()
    print("test")
