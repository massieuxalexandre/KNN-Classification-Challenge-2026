import numpy as np
import pandas as pd

class Data:
    def __init__(self, data, id_col, label_col):
        self.data = data 
        self.data_train = None 
        self.data_test = None 
        self.data_labels = None
        self.data_test_labels = None
        self.data_id = None
        self.label_col = label_col
        self.id_col = id_col


    def set_train_test(self, j):
        length = self.data.shape[0]
        # tab = [(0, 0.8), (0.1, 0.9), (0.2, 1)]
        # tab = [(0, 0.74), (0.16, 0.9), (0.26, 1)]
        # tab = [(0, 0.7), (0.15, 0.85), (0.3, 1)]
        tab = [(0, 0.9), (0.05, 0.95), (0.1, 1)]
        # tab = [(0, 0.85), (0.1, 0.95), (0.15, 1)]
        start = int(tab[j][0] * length)
        end = int(tab[j][1] * length)

        self.data_train = self.data.iloc[start:end, :].drop(columns=[self.id_col, self.label_col])
        self.data_labels = self.data.iloc[start:end][self.label_col]
        self.data_id = self.data.iloc[start:end][self.id_col]

        test_part1 = self.data.iloc[0:start, :]
        test_part2 = self.data.iloc[end:, :]
        self.data_test_labels = pd.concat([test_part1[self.label_col], test_part2[self.label_col]])
        self.data_test_id = pd.concat([test_part1[self.id_col], test_part2[self.id_col]])
        self.data_test = pd.concat([test_part1.drop(columns=[self.id_col, self.label_col]), test_part2.drop(columns=[self.id_col, self.label_col])])

        # moyenne = self.data_train.mean()
        # ecart_type = self.data_train.std()
        
        # self.data_train = (self.data_train - moyenne) / ecart_type
        # self.data_test = (self.data_test - moyenne) / ecart_type

        # Au lieu de moyenne et ecart_type, on utilise min et max
        val_min = self.data_train.min()
        val_max = self.data_train.max()

        self.data_train = (self.data_train - val_min) / (val_max - val_min)
        self.data_test = (self.data_test - val_min) / (val_max - val_min)




    def get_data_train(self):
        return self.data_train 
    
    def get_data_test(self):
        return self.data_test
    
    def get_data_labels(self):
        return self.data_labels
    
    def get_data_test_labels(self):
        return self.data_test_labels
    
    def get_label_col(self):
        return self.label_col
    
    def get_id_col(self):
        return self.id_col
