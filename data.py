from main import *
from knn import *
import numpy as np
import pandas as pd

class Data:
    def __init__(self, data, label_col):
        self.data = data 
        self.data_train = None 
        self.data_test = None 
        self.data_labels = None
        self.data_test_labels = None
        self.label_col = label_col
        self.cpt = 0


    def set_train_test(self):
        length = self.data.shape[0]
        tab = [(0, 0.8), (0.1, 0.9), (0.2, 1)]
        # tab = [(0, 0.7), (0.2, 0.9), (0.3, 1)]
        start = int(tab[self.cpt][0] * length)
        end = int(tab[self.cpt][1] * length)

        self.data_train = self.data.iloc[start:end, :].drop(columns=[self.label_col])
        self.data_labels = self.data.iloc[start:end][self.label_col]

        test_part1 = self.data.iloc[0:start, :]
        test_part2 = self.data.iloc[end:, :]
        self.data_test_labels = pd.concat([test_part1[self.label_col], test_part2[self.label_col]])
        self.data_test = pd.concat([test_part1.drop(columns=[self.label_col]), test_part2.drop(columns=[self.label_col])])

        self.cpt += 1


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

    



