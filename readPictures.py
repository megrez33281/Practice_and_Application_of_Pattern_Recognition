import os
from PIL import Image
import numpy as np

def readPictures(types_amount=40):
    pircure_root = os.getcwd()
    pircure_root = pircure_root + (r"\ORL3232")
    #讀取所有圖片，並記錄類別
    datas_train = []
    datas_test = []
    datas_train_type = []
    datas_test_type = []
    amounts = 0
    type_lists = os.listdir(pircure_root)
    #排序
    for i in range(len(type_lists)):
        bottom = len(type_lists)-1-i
        for j in range(bottom):
            if int(type_lists[j]) > int(type_lists[j+1]):
                type_lists[j], type_lists[j+1] = type_lists[j+1], type_lists[j]

    for data_type in type_lists:
        if data_type == "Non":
            continue 
        file_path = os.path.join(pircure_root, data_type)
        bmp_files = os.listdir(file_path)
        for file_root in [file for file in bmp_files if file.endswith('.bmp')]:
            picture_path = os.path.join(file_path, file_root)
            order = int(file_root.replace(".bmp", ""))
            img = Image.open(picture_path)
            img_array = np.array(img)
            #將圖片矩陣轉成向量
            cols = len(img_array) * len(img_array[0])
            img_vector = img_array.reshape((1, cols))[0]
            if order%2 == 1:
                datas_train.append(img_vector)
                datas_train_type.append(int(data_type))
            else:
                datas_test.append(img_vector)
                datas_test_type.append(int(data_type))
        amounts += 1
        if amounts >= types_amount:
            break

    return np.array(datas_train, dtype="float64"), datas_train_type,  np.array(datas_test, dtype="float64"), datas_test_type


def equal_array(arr1, arr2):
    #驗證一個一維陣列是否完全相同
    if len(arr1) != len(arr2):
        return False

    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False

    return True

def equal_Matrix(M1, M2):
    #驗證兩個matrix是否每一個row對方都有
    same = 1
    for row in M1:
        flag = 0
        for pic in M2:
            if equal_array(row, pic):
                flag = 1
                break
        if flag == 0:
            same = 0
            print(row)
        
    return same == 1


def getDimension(Matrix):
    return "(" + str(len(Matrix)) + ", " + str(len(Matrix[0])) + ")"


if __name__ == "__main__":
    
    train_data, train_type, test_data, test_type = readPictures()


