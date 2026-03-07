import numpy as np
import pandas as pd

from knn import *
from data import *


if __name__ == "__main__":
    df = pd.read_csv("sample_submission.csv")
    data = Data(df, "Label")

    for j in range(3):
        data.set_train_test()
        knn = Knn(4, data.get_data_train(), data.get_data_test(), data.get_data_labels())

        print("test n°", j+1, " : ", sep="")
        print("label reel - label trouve")
        # for i in range(len(data.get_data_test())):
        for i in range(5):
            print(data.get_data_test_labels().iloc[i], "-", knn.knn_prediction(data.get_data_test().iloc[i, :]))
        print()


    
