#自訂隱藏層+Entropy
import numpy as np
import math
import random
from Algfun import dlogsig, logsig, purelin, dpurelin, softmax, add_bias
from matplotlib import pyplot as plt
from readPictures import readPictures, equal_Matrix, equal_array
from printMatrix import printMatrix
from PCA import PCA, MaxMinNormalization


def weight_update(Input, weights, apha, bias, Target):
    SUMs = []
    A = [Input]
    DELTAs = []
    for weight in range(0, len(weights)):
      if weight == len(weights)-1:
        #最後一層，即輸出層，Aout
        sum = A[weight] @ weights[weight]
        SUMs.append(sum)
        A.append(softmax(sum + bias))
      else:
        sum = A[weight] @ weights[weight]
        SUMs.append(sum)
        A.append(logsig(sum + bias))

    #先計算最後的Delta
    Last_Delta = []
    #Aout = A[0][-1]
    #print("len", A[-1][0])
    for index in range(0, len(A[-1][0])):
      #注意負梯度
      if Target[0][index] == 1:
        Last_Delta.append(1 - A[-1][0][index] )
      else:
        Last_Delta.append(-1 * A[-1][0][index])
    Last_Delta = np.array([Last_Delta], dtype="complex")
    Last_Delta = Last_Delta.real
    #print(Last_Delta)

    DELTAs.append(Last_Delta)

    #計算其他Delta
    for delta in range(1, len(SUMs)):
      DELTAs.append((DELTAs[delta-1] @ np.transpose(weights[len(A)-delta-1]) * dlogsig(SUMs[len(SUMs)-delta-1], A[len(A)-delta-1])))
      #print(DELTAs[-1])
    for weight in range(0, len(weights)):
      weights[weight] = weights[weight] + apha * np.transpose(np.transpose(DELTAs[len(DELTAs)-weight-1]) @ A[weight])

    return weights


def classify(Aout):
    max = Aout[0]
    max_index = 0
    for num in range(0, len(Aout)):
        if Aout[num] > max:
            max = Aout[num]
            max_index = num
    return max_index+1

def train_model(Input, Target, test_data, test_Target, layers_and_neurons, learning_rate, bias, classify_amounts, epochs):
    apha = learning_rate
    epochs = epochs
    #注意矩陣維度輸入時的完整性
    weights = []
    neuron_every_layer = []
    neuron_every_layer.append([len(Input[0]), layers_and_neurons[0]])
    for layer in range(0, len(layers_and_neurons)-1):
        neuron_every_layer.append([layers_and_neurons[layer], layers_and_neurons[layer+1]])
    neuron_every_layer.append([layers_and_neurons[-1], classify_amounts])


    #權重初始化
    for n in neuron_every_layer:
        rows, cols = n
        w = []
        for row in range(0, rows):
            a_row = []
            for col in range(0, cols):
                a_row.append(random.uniform(-1, 1))
            w.append(a_row)
        w = np.array(w, dtype="float64")
        weights.append(w)


    LOSS_every_epoch = []
    for epoch in range(0, epochs):
        #一次epoch
        LOSS = [0]
        for data in range(len(Input)):
            weights = weight_update([Input[data]], weights, apha, bias, [Target[data]])
            Aout = logsig([Input[data]]@weights[0])
            for i in range(1, len(weights)-1):
                Aout = logsig(Aout@weights[i])

            Aout = softmax(Aout@weights[-1])
            #print(np.sum(Aout))
            for classify_index in range(0, len(LOSS)):
                target_index = 0
                for i in range(0, len(Target[data])):
                    if Target[data][i] == 1:
                        target_index = i
                        break
                LOSS[classify_index] += -1*math.log(Aout[0][target_index].real)



        LOSS = (sum(LOSS))/(len(train_data))
        print("Epoch {:03}".format(epoch+1) + ":", "LOSS = {:.16f}".format(LOSS))
        LOSS_every_epoch.append(LOSS)

    #Test Data
    correct = 0
    outputs = []
    for data in range(len(test_data)):
        Aout = logsig([test_data[data]]@weights[0])
        for i in range(1, len(weights)-1):
            Aout = logsig(Aout@weights[i])
        Aout = softmax(Aout@weights[-1])
        output = [Aout[0]]
        res = classify(Aout[0])
        if res == classify(test_Target[data]):
            output.append("True")
            correct += 1
        else:
            output.append("False")
        outputs.append(output)
    print(correct, len(test_data))
    success = correct/len(test_data)

    return success, LOSS_every_epoch, outputs

def read_Input_Target(features):

    ######似乎Train data和Test data在選取特徵值時選到了不同順序的特徵
    train_data, train_type, test_data, test_type = readPictures()
    PCA_train_data, PCA_test_data = PCA(train_data, test_data, features)
    train_data_Max_Min, test_data_Max_Min = MaxMinNormalization(PCA_train_data, PCA_test_data)
    train_Target = []
    test_Target = []
    classify_amounts = np.max(train_type)
    #取得train data的target
    for dataType in train_type:
        now_target = []
        for i in range(1, classify_amounts+1):
            if i == dataType:
               now_target.append(1)
            else:
               now_target.append(0)
        now_target = np.array(now_target, dtype="float64")
        train_Target.append(now_target)
    train_Target = np.array(train_Target, dtype="float64")


    #取得test data的target
    for dataType in test_type:
        now_target = []
        for i in range(1, classify_amounts+1):
            if i == dataType:
               now_target.append(1)
            else:
               now_target.append(0)
        now_target = np.array(now_target, dtype="float64")
        test_Target.append(now_target)
    test_Target = np.array(test_Target, dtype="float64")

    return train_data_Max_Min, train_Target, test_data_Max_Min, test_Target

if __name__ == '__main__':
    layers_and_neurons = [150]
    learning_rate = 0.05
    bias = 0.1
    classify_amounts = 40
    epochs = 500
    features = 65
    train_data, train_Target, test_data, test_Target = read_Input_Target(features)
    success, RMSE_every_epoch, outputs = train_model(train_data, train_Target, test_data, test_Target, layers_and_neurons, learning_rate, bias, classify_amounts, epochs)

    '''
    #輸出test data的輸出結果
    print()
    for output in range(0, len(outputs)):
        print("資料" + "{:03}".format(output+76) + ":", end=' ')
        for index in outputs[output][0]:
            print("{:20.16f}".format(index), end=' ')
        print("Target:", int(classify(test_Target[output])), "Hit:", outputs[output][1])'''

    print(success)

    #劃出RMSE-Epochs圖
    plt.rcParams["figure.figsize"] = (6,6)
    plt.plot(RMSE_every_epoch,color = 'b')
    plt.title('LOSS and Epochs')
    plt.ylabel('LOSS')
    plt.xlabel('Epochs')
    plt.show()


