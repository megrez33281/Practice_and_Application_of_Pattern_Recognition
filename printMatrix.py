import numpy as np
import random


def printMatrix(nparray):
    #漂亮的印出matrix
    for row in nparray:
        for col in row:
            print("{:6.2f}".format(col), end='    ')
        print()
    return


if __name__ == "__main__":


    a_list = []
    for i in range(10):
        temp = []
        for j in range(10):
            temp.append(random.uniform(-1, 1))
        a_list.append(temp)
    printMatrix(a_list)