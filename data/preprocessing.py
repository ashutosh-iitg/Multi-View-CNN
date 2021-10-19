import os
import re
import json
import numpy as np
import pandas as pd

class PreProcessing():
    def __init__(self, csv_dir) -> None:
        dict_path = "config/label_dict.json"
        if os.path.isfile(dict_path):
            with open(dict_path) as f:
                self.label_dict = json.load(f)
        else:
            df = pd.read_csv(os.path.join(csv_dir, "train_data.csv"))
            self.label_dict = self.create_dict(df)
        
        self.train_df = self.create_encoded_df(os.path.join(csv_dir, "train_data.csv"))
        self.test_df = self.create_encoded_df(os.path.join(csv_dir, "test_data.csv"))

    def create_dict(self, df):
        label_unique = np.unique(df.Genus)
        label_dict = {label: i for i, label in enumerate(label_unique)}

        with open("config/label_dict.json", 'w') as f:
            json.dump(label_dict, f, indent=4)
        
        return label_dict

    def create_encoded_df(self, csv_path):
        df = pd.read_csv(csv_path)
        label_list = df.Genus.tolist()
        label_encoded = np.array([self.label_dict[label] for label in label_list])
        df.Genus = label_encoded
        return df