import numpy as np
import pandas as pd

from knn import *
from data import *


if __name__ == "__main__":
    # df = pd.read_csv("sample_submission.csv")
    df = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    # df = pd.read_csv("data_iris.txt")
    data = Data(df, "Id", "Label")
    # data_test = Data(df_test, "Label")
    # data = Data(df, "Iris-setosa")
    k = 3
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
    # knn = Knn(k, df.iloc[:, 1:], df_test.iloc[:, 1:], df["Label"])
    # for i in range(len(df_test)):
    #     guess[df_test.iloc[i, 0]] = knn.knn_prediction(df_test.iloc[i, 1:])

    # df_final = pd.DataFrame(list(guess.items()), columns=["Id", "Label"])
    # print(df_final)
    # df_final.to_csv("guess_final.csv", index=False)

    # print("test final (Soumission Kaggle) :")
    

    X_train_final = df.drop(columns=["Id", "Label"])
    y_train_final = df["Label"]
    
    X_test_final = df_test.drop(columns=["Id"])

    val_min = X_train_final.min()
    val_max = X_train_final.max()
    
    X_train_final = (X_train_final - val_min) / (val_max - val_min)
    X_test_final = (X_test_final - val_min) / (val_max - val_min)

    knn_final = Knn(k, X_train_final, X_test_final, y_train_final)
    
    guess = dict()
    for i in range(len(df_test)):
        guess[df_test.iloc[i, 0]] = knn_final.knn_prediction(X_test_final.iloc[i, :])

    df_final = pd.DataFrame(list(guess.items()), columns=["Id", "Label"])
    print(df_final.head())
    df_final.to_csv("guess_final.csv", index=False)