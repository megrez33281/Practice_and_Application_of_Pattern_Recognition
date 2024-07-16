from scipy.io import loadmat
import numpy as np

def get_ORL_RawData():
    #ORL_RawData讀取.mat檔
    mat = loadmat('ORL_RawData.mat')
    ORLrawdataTrain = mat['ORLrawdataTrain']
    ORLrawdataTest = mat['ORLrawdataTest']
    ORLrawdata = []
    for row in ORLrawdataTrain:
        ORLrawdata.append(np.array(row, dtype="float64"))
    for row in ORLrawdataTest:
        ORLrawdata.append(np.array(row, dtype="float64"))

    ORLrawdata = np.array(ORLrawdata, dtype="float64")
    return ORLrawdata

def get_ORL_PCA_MaxMin_Data():
    #ORL_PCA_MaxMin_Data讀取.mat檔
    mat = loadmat('ORL_PCA_MaxMin_Data.mat')
    TrainORL = mat['TrainORL']
    TestORL = mat['TestORL']
    ORLdata = []
    for row in TrainORL:
        ORLdata.append(row)
    for row in TestORL:
        ORLdata.append(row)

    ORLdata = np.array(ORLdata, dtype="float64")
    #四捨五入至小數點第4位
    ORLdata = np.round(ORLdata, 4)
    return ORLdata


if __name__ == "__main__":

    ORL_MaxMin_data = get_ORL_PCA_MaxMin_Data()
    print(len(ORL_MaxMin_data[0]))
    ORL_RawData = get_ORL_RawData()