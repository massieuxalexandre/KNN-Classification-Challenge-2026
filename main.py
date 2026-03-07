import numpy as np
import pandas as pd

from knn import *
from data import *


if __name__ == "__main__":
    df = pd.read_csv("data_iris.txt")
    data = Data(df, "Iris-setosa")

    for j in range(3):
        data.set_train_test()
        knn = Knn(3, data.get_data_train(), data.get_data_test(), data.get_data_labels())

        print("test n°", j+1, " : ", sep="")
        print("label reel - label trouve")
        for i in range(len(data.get_data_test())):
            print(data.get_data_test().iloc[i, 4], "-", knn.knn_prediction(data.get_data_test().iloc[i, :]))
        print()


    
