import numpy as np
from readPictures import readPictures
from printMatrix import printMatrix
from test_readPictures import get_ORL_PCA_MaxMin_Data
from readPictures import equal_Matrix, getDimension

def getMean(Matrix):
    means = []
    for col in range(len(Matrix[0])):
        means.append(np.mean(Matrix[:, col]))
    return means

def MatrixShift(Matrix, means):
    for col in range(len(means)):
        Matrix[:, col] -= means[col]
    return Matrix

def getEigen(Matrix, MeanOfTrain):
    #計算矩陣的eigenvalues和eigenvectors
    Matrix_Normalize = []
    for col in range(len(MeanOfTrain)):
        #取平均
        means = MeanOfTrain[col]
        #正規化
        Matrix_Normalize.append(Matrix[:, col] - means)

    #計算共變異數
    CovM = Matrix_Normalize @ np.transpose(Matrix_Normalize)
    #計算特徵值，特徵向量
    eigenvalues, eigenvectors = np.linalg.eig(CovM)
    return eigenvalues, eigenvectors


def MaxMinNormalization(PCA_train_data, PCA_test_data):
    train_data_Max_Min = []
    test_data_Max_Min = []
    for i in range(len(PCA_train_data[0])):
        train_column = PCA_train_data[:,i]
        test_column = PCA_test_data[:,i]
        #使用train data的最大最小值
        max_num = max(train_column)
        min_num = min(train_column)
        train_data_Max_Min.append((train_column - min_num)/(max_num - min_num))
        test_data_Max_Min.append((test_column - min_num)/(max_num - min_num))
    train_data_Max_Min = np.transpose(train_data_Max_Min)
    test_data_Max_Min = np.transpose(test_data_Max_Min)
    #MaxMinNormalization = np.array(MaxMinNormalization, dtype="float64")
    return train_data_Max_Min, test_data_Max_Min


def PCA(data_train_Matrix, data_test_Matrix, features):
    #features為要保留的特徵(維度)之數量
    MeanOfTrain = getMean(data_train_Matrix)
    eigenvalues, eigenvectors = getEigen(data_train_Matrix, MeanOfTrain)
    
    #資料平移
    train_shift = MatrixShift(data_train_Matrix, MeanOfTrain)
    test_shift = MatrixShift(data_test_Matrix, MeanOfTrain)

    #取得eigenvaluse由大到小排序的索引
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_indices[:features]]
    PCA_train_data = train_shift @ selected_eigenvectors
    PCA_test_data = test_shift @ selected_eigenvectors

    return PCA_train_data, PCA_test_data


if __name__ == "__main__":

    train_data, train_type, test_data, test_type = readPictures()
    PCA_train_data, PCA_test_data = PCA(train_data, test_data, 65)
    train_data_Max_Min, test_data_Max_Min = MaxMinNormalization(PCA_train_data, PCA_test_data)
    train_data_Max_Min = np.around(train_data_Max_Min, 4)
    train, test = get_ORL_PCA_MaxMin_Data()
    train = np.around(train, 4)
    test = np.around(test, 4)
    print(equal_Matrix(train_data_Max_Min, train))




    


    
    
