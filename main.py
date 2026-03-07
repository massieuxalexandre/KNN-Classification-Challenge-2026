import numpy as np
import pandas as pd

from knn import *
from data import *


if __name__ == "__main__":
    df = pd.read_csv("sample_submission.csv")
    # df = pd.read_csv("data_iris.txt")
    data = Data(df, "Label")
    # data = Data(df, "Iris-setosa")
    k = 5
    

    for j in range(3):
        good = 0
        total = 0

        data.set_train_test()
        knn = Knn(k, data.get_data_train(), data.get_data_test(), data.get_data_labels())

        print("test n°", j+1, " : ", sep="")
        print("label reel - label trouve")
        for i in range(len(data.get_data_test())):
        # for i in range(5):
            print(data.get_data_test_labels().iloc[i], "-", knn.knn_prediction(data.get_data_test().iloc[i, :]))
            total += 1
            if (data.get_data_test_labels().iloc[i] == knn.knn_prediction(data.get_data_test().iloc[i, :])):
                good += 1
        accuracy = good / total
        print("accuracy test n°", j, " : ", accuracy*100, " %", sep="")
        print()


    
