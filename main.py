import numpy as np
import pandas as pd

from knn import *
from data import *


if __name__ == "__main__":
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    data = Data(df_train, "Id", "Label")
    k = 1
    accuracy = []
    print("3 test locaux (differents splits) :")
    for j in range(3):
        good = 0
        total = 0
        guess = dict()

        data.set_train_test(j)
        knn = Knn(k, data.get_data_train(), data.get_data_test(), data.get_data_labels())

        print("test n°", j+1, " : ", sep="")
        # print("label reel - label trouve")
        for i in range(len(data.get_data_test())):
        # for i in range(5):
            # print(data.get_data_test_labels().iloc[i], "-", knn.knn_prediction(data.get_data_test().iloc[i]))
            total += 1
            if (data.get_data_test_labels().iloc[i] == knn.knn_prediction(data.get_data_test().iloc[i])):
                good += 1
        accuracy.append(good / total)
        print("accuracy test n°", j+1, " : ", accuracy[j]*100, " %", sep="")
        print()

    accuracy_mean = sum(accuracy)/len(accuracy)
    print("moyenne accuracy : ", accuracy_mean*100, " %", sep="")
    print()



    # test final en s'entrainant sur tout le train
    # print("test final :")
    # data_final = Data_final(df_train, df_test, "Id", "Label")
    # data_final.set_train_test_final()
    
    # guess_final = dict()
    # knn = Knn(k, data_final.get_data_train(), data_final.get_data_test(), data_final.get_data_labels())
    # for i in range(len(data_final.get_data_test())):
    #     guess_final[data_final.get_data_test_id().iloc[i]] = knn.knn_prediction(data_final.get_data_test().iloc[i])

    # df_final = pd.DataFrame(list(guess_final.items()), columns=["Id", "Label"])
    # print(df_final)
    # df_final.to_csv("guess_final.csv", index=False)

