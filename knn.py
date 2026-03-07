from main import *
from data import *
import numpy as np
import pandas as pd

class Knn:
    def __init__(self, k, data_train, data_test, data_labels):
        self.k = k
        self.data_train = data_train
        self.data_test = data_test
        self.data_labels = data_labels

    def euclidean_distance(self, x_1, x_2):
        return np.sqrt(np.sum((np.array(x_1[:-1]) - np.array(x_2[:-1]))**2))
    
    def manhattan_distance(self, x_1, x_2):
        return np.sum(np.abs(np.array(x_1[:-1]) - np.array(x_2[:-1])))
    
    
    def knn_prediction(self, point_test):
        distances = []

        for i in range(len(self.data_train)):
            dist = self.euclidean_distance(point_test, self.data_train.iloc[i])
            # dist = self.manhattan_distance(point_test, self.data_train.iloc[i])
            distances.append((dist, self.data_labels.iloc[i]))

        distances.sort(key=lambda x: x[0])
        top_k = []
        for (distance, label) in distances[:self.k]:
            top_k.append(label)

        return max(set(top_k), key=top_k.count)
