import numpy as np
import matplotlib.pyplot as plt
from readPictures import readPictures
from printMatrix import printMatrix
from test_readPictures import get_ORL_PCA_MaxMin_Data
from readPictures import equal_Matrix, getDimension, equal_array
from PCA import PCA, getMean, MatrixShift

def getCovM(Matrix):
    #計算矩陣的共變異數
    Matrix_Normalize = []
    for col in range(len(Matrix[0])):
        #取平均
        means = np.mean(Matrix[:, col])
        #正規化
        Matrix_Normalize.append(Matrix[:, col] - means)

    #計算共變異數
    CovM = np.dot(Matrix_Normalize, np.transpose(Matrix_Normalize))
    return CovM

def getEigen(Matrix):
    #計算矩陣的eigenvalues和eigenvectors
    #計算特徵值，特徵向量
    eigenvalues, eigenvectors = np.linalg.eig(Matrix)
    return eigenvalues, eigenvectors

def LDA(data_train_Matrix, train_type, data_test_Matrix, features):
    type_amounts = np.max(train_type)

    #將不同類別的資料分開
    Within = []
    for types in range(type_amounts):
        Within.append([])

    for data in range(len(data_train_Matrix)):
        Within[train_type[data]-1].append(data_train_Matrix[data])
    Within = np.array(Within, dtype="complex")  #確保數據為實數
    Within = Within.real

    #計算不同類別各自的CovM
    Within_CovMs = []
    for types in range(type_amounts):
        Within_CovMs.append(getCovM(Within[types]))
    Within_CovMs = np.array(Within_CovMs, dtype="float64")

    #加總所有的CovM
    Within_CovMs_Sum = np.zeros((len(Within_CovMs[0]), len(Within_CovMs[0][0])))
    for CovM in Within_CovMs:
        Within_CovMs_Sum += CovM
 
    #計算Between的CovM
    Between_CovM = getCovM(data_train_Matrix)

    #計算最終的inv(Within)*Between
    Final_Matrix = np.linalg.inv(Within_CovMs_Sum) @ Between_CovM
    eigenvalues, eigenvectors = getEigen(Final_Matrix)
    #取得eigenvaluse由大到小排序的索引
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_indices[:features]]
    LDA_train_data = data_train_Matrix @ selected_eigenvectors
    LDA_test_data = data_test_Matrix @ selected_eigenvectors

    return LDA_train_data, LDA_test_data, eigenvalues, eigenvectors

def getDistance(arr1, arr2):
    adds = 0
    for num in range(len(arr1)):
        adds += (arr1[num]-arr2[num])**2
    return adds

def classify(LDA_test_arr, LDA_train_data, train_type):
    nearest = getDistance(LDA_test_arr, LDA_train_data[0])
    nearest_type = train_type[0]
    for row in range(len(LDA_train_data)):
        dis = getDistance(LDA_test_arr, LDA_train_data[row])
        if dis < nearest:
            nearest = dis
            nearest_type = train_type[row]
    return nearest_type

def plot_lda_3d(LDA_train_data, train_type, LDA_test_data, test_type):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for each type
    unique_types = np.unique(train_type)
    colors = plt.get_cmap('tab20', len(unique_types))  # Use 'tab20' colormap
    color_map = {t: colors(i % 20) for i, t in enumerate(unique_types)}

    # Plot training data
    for t in unique_types:
        indices = np.where(train_type == t)
        ax.scatter(LDA_train_data[indices, 0].real, LDA_train_data[indices, 1].real, LDA_train_data[indices, 2].real, 
                   label=f'Train Type {t}', color=color_map[t])

    # Plot test data
    for t in unique_types:
        indices = np.where(test_type == t)
        ax.scatter(LDA_test_data[indices, 0].real, LDA_test_data[indices, 1].real, LDA_test_data[indices, 2].real, 
                   marker='x', label=f'Test Type {t}', color=color_map[t])

    ax.set_xlabel('LDA Feature 1')
    ax.set_ylabel('LDA Feature 2')
    ax.set_zlabel('LDA Feature 3')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    #讀取資料
    train_data, train_type, test_data, test_type = readPictures(20)

    #完成PCA
    PCA_train_data, PCA_test_data = PCA(train_data, test_data, 65)
    #資料平移
    MeanOfTrain = getMean(PCA_train_data)
    LDA_train_data, LDA_test_data, eigenvalues, eigenvectors = LDA(PCA_train_data, train_type, PCA_test_data, 20)

    #分類
    success = 0
    for row in range(len(LDA_test_data)):
        predict_test_data_type = classify(LDA_test_data[row], LDA_train_data, train_type)
        expect_type = test_type[row]
        if predict_test_data_type == expect_type:
            success += 1

    print("accuracy:", success/len(LDA_test_data))

    #取前3個最大的特徵
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top3_indices = sorted_indices[:3]

    LDA_train_data_3d = LDA_train_data[:, top3_indices]
    LDA_test_data_3d = LDA_test_data[:, top3_indices]

    #3D視覺化
    plot_lda_3d(LDA_train_data_3d, train_type, LDA_test_data_3d, test_type)