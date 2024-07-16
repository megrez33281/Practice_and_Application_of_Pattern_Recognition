#自訂隱藏層+Entropy
import numpy as np
import math
import random
from Algfun import dlogsig, logsig, purelin, dpurelin, softmax, add_bias
from matplotlib import pyplot as plt
from readPictures import readPictures, equal_Matrix, equal_array
from printMatrix import printMatrix
from PCA import PCA, MaxMinNormalization


def weight_update(Input, weights, apha, bias, Target, epoch_lambda, Marquardt):
    def propogate(weights, bias):
        all_SUMs = []   #所有訓練資料的各層的SUM紀錄
        all_A = []  #所有訓練資料的各層的輸出紀錄
        for train_input in Input:
            SUMs = []
            A = [[train_input]]   #視為第0層輸入
            for weight in range(0, len(weights)):
                if weight == len(weights)-1:
                    #最後一層，即輸出層，Aout
                    sum = A[weight] @ weights[weight]
                    SUMs.append(sum)
                    A.append(softmax(sum + [bias[weight]]))
                else:
                    sum = A[weight] @ weights[weight]
                    SUMs.append(sum)
                    A.append(logsig(sum + [bias[weight]]))
            all_SUMs.append(SUMs)
            all_A.append(A)
        return all_SUMs, all_A

    def getPerformanceIndex(all_A):
        performance_index = 0
        #紀錄每一筆資料的error
        Error = []
        for aout in range(len(all_A)):
            #計算每筆資料的(T-A)
            data_out = all_A[aout][-1]
            data_target = Target[aout]
            err = (data_target - data_out)
            Error.append(err[0])
            #計算(T-A)*transpose(T-A)
            performance_index += ((err)@np.transpose(err))[0][0].real
        return performance_index, Error
    
    def getJecobianMatrix(all_A, all_SUMs, Error):
        #初始化並倒傳遞Marquardt sensitivities（delta）
        #計算每一筆資料的每一層的Marquardt sensitivities
        all_Deltas = [] #每一筆資料的每一層Delta，以層為row，資料筆數為col
        for i in range(len(all_A[0])-1):
           all_Deltas.append([])
        for index  in range(len(all_A)):
            DELTAs = [] #紀錄該筆資料每一層的Delta
            A = all_A[index]
            SUMs = all_SUMs[index]
            #先計算最後的Delta
            Last_Delta = np.identity(len(A[-1][0]))
            for index in range(0, len(A[-1][0])):
                #注意負梯度
                #S[h][i]
                if Target[0][index] == 1:
                    Last_Delta[index][index] = (1 - A[-1][0][index].real )
                else:
                    Last_Delta[index][index] = (-1 * A[-1][0][index].real)
            Last_Delta = np.array(Last_Delta, dtype="complex")
            Last_Delta = Last_Delta.real
            DELTAs.append(Last_Delta)
            #計算其他層Delta
            for delta in range(1, len(SUMs)):
                other_delta = np.identity(len(A[len(A)-delta-1][0]))
                dtransfer = dlogsig(SUMs[len(SUMs)-delta-1], A[len(A)-delta-1])
                for ele in range(len(dtransfer[0])):
                    other_delta[ele][ele] = dtransfer[0][ele].real
                DELTAs.append(other_delta@weights[len(A)-delta-1]@DELTAs[delta-1])
            #將該筆資料的delta放入all_Deltas
            for delta in range(len(DELTAs)):
                #把每一層的delta放到all_Delta對應層
                #all_Deltas = [[train_data1_layer1_delta, train_data2_layer1_delta,train_data3_layer1_delta......]......]
                #print("delta before=",delta, len(all_Deltas[delta]))
                for dels in range(len(DELTAs[len(DELTAs)-1-delta][0])):
                    all_Deltas[delta].append(DELTAs[len(DELTAs)-1-delta][:,dels])
                #print("delta after=",delta, len(all_Deltas[delta]))

        for data in range(len(all_Deltas)):
            all_Deltas[data] = np.transpose(all_Deltas[data])
            #all_Deltas = [layer1, layer2, layer3......]
            #layer1 = [train_data1_delta, train_data2_delta, train_data3_delta......]

        #計算Jacobian matrix
        J = []
        error_line = []
        for q in range(len(Error)):
            for k in range(len(Error[q])):
                j_row = []
                error_line.append(Error[q][k])
                #每一個error會對各層的各個error、bias做偏導數

                for m in range(len(weights)):
                    #計算對權重的偏微
                    h = ((q-1)*len(Error[q])+k)
                    for j in range(len(weights[m][0])):
                        for i in range(len(weights[m])):   
                            j_row.append(all_Deltas[m][j][h]*all_A[q][m][0][i])

                    #計算對bias的偏微
                    for the_bias in range(len(bias[m])):
                        j_row.append(all_Deltas[m][the_bias][h])
                J.append(j_row)

        J = np.array(J, dtype="complex")
        error_line = np.transpose([error_line])
        return J.real, error_line


    def try_renew_weight(Jacobian):
        j_mul = np.transpose(Jacobian)@Jacobian + epoch_lambda*np.identity(len(np.transpose(Jacobian)))
        #weight_delta
        weight_delta = -1*np.linalg.inv(j_mul)@np.transpose(Jacobian)@error_line
        new_weights = weights.copy()
        new_bias = bias.copy()
        #每層的weight
        renew_index = 0
        for m in range(len(weights)):
            #每一組weight
            for j in range(len(weights[m][0])):
                #每一個weight
                for i in range(len(weights[m])):
                    new_weights[m][i][j] += weight_delta[renew_index][0].real
                    renew_index +=1
                    
            #計算新bias
            for the_bias in range(len(bias[m])):
                new_bias[m][the_bias]  += weight_delta[renew_index][0].real
                renew_index += 1
        return new_weights, new_bias
    
    def renew_weight(Jacobian):
        j_mul = np.transpose(Jacobian)@Jacobian + epoch_lambda*np.identity(len(np.transpose(Jacobian)))
        #weight_delta
        weight_delta = -1*np.linalg.inv(j_mul)@np.transpose(Jacobian)@error_line
        #每層的weight
        renew_index = 0
        for m in range(len(weights)):
            #每一組weight
            for j in range(len(weights[m][0])):
                #每一個weight
                for i in range(len(weights[m])):
                    weights[m][i][j] += apha*weight_delta[renew_index][0].real
                    renew_index +=1
                    
            #計算新bias
            for the_bias in range(len(bias[m])):
                bias[m][the_bias]  += apha*weight_delta[renew_index][0].real
                renew_index += 1  
    
    #進行正傳遞，計算所有資料的輸出
    all_SUMs, all_A = propogate(weights, bias)
    #計算Performance index F(X)用以之後計算新的lambda
    performance_index, Error = getPerformanceIndex(all_A)
    new_performance_index = performance_index + 1
    #計算Jacobian Matrix
    Jacobian, error_line = getJecobianMatrix(all_A, all_SUMs, Error)
    
    #嘗試更新權重
    renew_weight(Jacobian)
    
    #重新計算所有資料的輸出
    all_SUMs, all_A = propogate(weights, bias)
    new_performance_index, Error = getPerformanceIndex(all_A)
    
    #根據新的Performance index調整lambda
    if new_performance_index > performance_index:
        epoch_lambda *= Marquardt
    else:
        epoch_lambda /= Marquardt
    
    return weights, bias, epoch_lambda


def classify(Aout):
    max = Aout[0]
    max_index = 0
    for num in range(0, len(Aout)):
        if Aout[num] > max:
            max = Aout[num]
            max_index = num
    return max_index+1

def shuffling(data, data_type):
    #生成一個隨機排列的索引
    indices = np.random.permutation(len(data))
    
    #根據隨機索引打亂 data 和 data_type
    shuffled_data = data[indices]
    shuffled_type = data_type[indices]
    
    return shuffled_data, shuffled_type

def train_model(Input, Target, test_data, test_Target, layers_and_neurons, learning_rate, bias, classify_amounts, epochs, train_lambda, Marquardt):
    apha = learning_rate
    epochs = epochs
    #注意矩陣維度輸入時的完整性
    weights = []
    
    neuron_every_layer = []
    neuron_every_layer.append([len(Input[0]), layers_and_neurons[0]])
    for layer in range(0, len(layers_and_neurons)-1):
        neuron_every_layer.append([layers_and_neurons[layer], layers_and_neurons[layer+1]])

    neuron_every_layer.append([layers_and_neurons[-1], classify_amounts])
    all_bias = []
    for layer in layers_and_neurons:
        layer_bias = []
        for b in range(layer):
            layer_bias.append(bias)
        all_bias.append(np.array(layer_bias, dtype="float64"))
    layer_bias = []
    for b in range(classify_amounts):
        layer_bias.append(bias)
    all_bias.append(np.array(layer_bias, dtype="float64"))

    #權重初始化
    for n in neuron_every_layer:
        rows, cols = n
        w = np.random.randn(rows, cols) * np.sqrt(2 / (rows + cols))  #Xavier初始化
        weights.append(w)

    LOSS_every_epoch = []
    epoch_lambda = train_lambda
    for epoch in range(0, epochs):
        #一次epoch
        weights, bias, epoch_lambda = weight_update(Input, weights, apha, all_bias, Target, epoch_lambda, Marquardt)
        #計算權重更新後的結果
        total_loss = 0
        for data in range(len(Input)):
            Aout = logsig([Input[data]] @ weights[0])
            for i in range(1, len(weights)-1):
                Aout = logsig(Aout @ weights[i])
            Aout = softmax(Aout @ weights[-1])
            target_index = np.argmax(Target[data])
            clipped_value = np.clip(Aout[0][target_index].real, 1e-15, 1)  #對輸入值進行裁剪
            total_loss += -1 * math.log(clipped_value)
        
        LOSS = total_loss / len(Input)
        LOSS_every_epoch.append(LOSS)
        print(f"Epoch {epoch+1:03}: LOSS = {LOSS:.16f}")

        #Test Data
        correct = 0
        outputs = []
        for data in range(len(test_data)):
            Aout = logsig([test_data[data]] @ weights[0])
            for i in range(1, len(weights)-1):
                Aout = logsig(Aout @ weights[i])
            Aout = softmax(Aout @ weights[-1])
            output = [Aout[0]]
            res = classify(Aout[0])
            if res == classify(test_Target[data]):
                output.append("True")
                correct += 1
            else:
                output.append("False")
            outputs.append(output)
        success = correct / len(test_data)
        print(f"Test Data: {correct}/{len(test_data)} correct, success rate = {success:.2%}")

        #隨機打亂訓練數據
        Input, Target = shuffling(Input, Target)

    return success, LOSS_every_epoch, outputs

def read_Input_Target(features, total_amounts):

    ######似乎Train data和Test data在選取特徵值時選到了不同順序的特徵
    train_data, train_type, test_data, test_type = readPictures(total_amounts)
    PCA_train_data, PCA_test_data = PCA(train_data, test_data, features)
    train_data_Max_Min, test_data_Max_Min = MaxMinNormalization(PCA_train_data, PCA_test_data)
    train_Target = []
    test_Target = []
    classify_amounts = total_amounts
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
    layers_and_neurons = [70]
    Marquardt = 2 #每次更新完權重後，依該次的performance index將train_lambda乘上或除上該數
    train_lambda = 0.3
    learning_rate = 0.8
    bias = 0.05
    classify_amounts = 10
    epochs = 10
    features = 40
    train_data, train_Target, test_data, test_Target = read_Input_Target(features, classify_amounts)
    success, LOSS_every_epoch, outputs = train_model(train_data, train_Target, test_data, test_Target, layers_and_neurons, learning_rate, bias, classify_amounts, epochs, train_lambda, Marquardt)

    print(f"Final success rate: {success:.2%}")

    #劃出LOSS-Epochs圖
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.plot(LOSS_every_epoch, color='b')
    plt.title('LOSS and Epochs')
    plt.ylabel('LOSS')
    plt.xlabel('Epochs')
    plt.show()