import numpy as np

def logsig(n):
    n = np.clip(n, -500, 500)  # 對輸入值進行裁剪
    return 1 / (1 + np.exp(-n))

def dlogsig(n, a):
    return a * (1 - a)

def purelin(n):
    return n

def dpurelin(n, a):
    return np.ones_like(n)

def softmax(n):
    e_n = np.exp(n - np.max(n))
    return e_n / e_n.sum(axis=1, keepdims=True)

def add_bias(n):
    return np.hstack([n, np.ones((n.shape[0], 1))])