import numpy as np
import pandas as pd

from knn import *
from data import *


if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    data = Data(df, "Id", "Label")
    data_final = Data_final(df, df_test, "Id", "Label")
    k = 1
    accuracy = []
    print("3 test pour trouver les meilleurs settings :")
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

    total_final = 0
    good_final = 0
    best = accuracy.index(max(accuracy))
    print("reentrainemnt avec les meilleurs settings :")
    data.set_train_test(best)
    knn = Knn(k, data.get_data_train(), data.get_data_test(), data.get_data_labels())
    for i in range(len(data.get_data_test())):
        total_final += 1
        if (data.get_data_test_labels().iloc[i] == knn.knn_prediction(data.get_data_test().iloc[i])):
            good_final += 1
    accuracy_final = good_final / total_final
    print("accuracy : ", accuracy_final*100, " %", sep="")
    print()

    print("test final :")
    # test final en s'entrainant sur tout le train
    # data_final.set_train_test_final()
    # knn = Knn(k, data_final.get_data_train(), data_final.get_data_test(), data_final.get_data_labels())
    # for i in range(len(data_final.get_data_test())):
    #     guess[data_final.get_data_test_id().iloc[i]] = knn.knn_prediction(data_final.get_data_test().iloc[i])

    # df_final = pd.DataFrame(list(guess.items()), columns=["Id", "Label"])
    # print(df_final)
    # df_final.to_csv("guess_final.csv", index=False)

    # test final en s'entrainant sur 90% du train
    df_90 = df.sample(frac=0.9, random_state=42)
    data_final_90 = Data_final(df_90, df_test, "Id", "Label")
    data_final_90.set_train_test_final()
    knn = Knn(k, data_final_90.get_data_train(), data_final_90.get_data_test(), data_final_90.get_data_labels())
    guess_90 = dict()
    for i in range(len(data_final_90.get_data_test())):
        guess_90[data_final_90.get_data_test_id().iloc[i]] = knn.knn_prediction(data_final_90.get_data_test().iloc[i])
    
    df_final_90 = pd.DataFrame(list(guess_90.items()), columns=["Id", "Label"])
    print(df_final_90)
    df_final_90.to_csv("guess_final_90.csv", index=False)

