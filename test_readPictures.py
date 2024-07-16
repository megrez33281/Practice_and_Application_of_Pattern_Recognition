import pytest
from scipy.io import loadmat
import numpy as np
from readPictures import equal_array, equal_Matrix, readPictures

def get_ORL_RawData():
    #ORL_RawData讀取.mat檔
    mat = loadmat('ORL_RawData.mat')
    ORLrawdataTrain = mat['ORLrawdataTrain']
    ORLrawdataTest = mat['ORLrawdataTest']
    return np.array(ORLrawdataTrain, dtype="float64"), np.array(ORLrawdataTest, dtype="float64")

def get_ORL_PCA_MaxMin_Data():
    #ORL_PCA_MaxMin_Data讀取.mat檔
    mat = loadmat('ORL_PCA_MaxMin_Data.mat')
    TrainORL = mat['TrainORL']
    TestORL = mat['TestORL']
    return np.array(TrainORL, dtype="float64"), np.array(TestORL, dtype="float64")


def test_numpy_equal_array():
    test_data = [np.array([]), np.array([])]
    result  = equal_array(test_data[0], test_data[1])
    except_result = True
    assert (result == except_result)

def test_long_true_equal_array():
    test_data = [np.array(list(range(10000))), np.array(list(range(10000)))]
    result  = equal_array(test_data[0], test_data[1])
    except_result = True
    assert (result == except_result)

def test_long_false_equal_array():
    test_data = [np.array(list(range(10000))), np.array(list(range(10000, 1, -1)))]
    result  = equal_array(test_data[0], test_data[1])
    except_result = False
    assert (result == except_result)

def test_equal_matrix():
    test_data = [np.array([[], []]), np.array([[], []])]
    result  = equal_array(test_data[0], test_data[1])
    except_result = True
    assert (result == except_result)


def test_readPictures():
    train_data, train_type, test_data, test_type = readPictures()
    mat_train, mat_test = get_ORL_RawData()
    assert equal_Matrix(train_data, mat_train)
    assert equal_Matrix(test_data, mat_test)