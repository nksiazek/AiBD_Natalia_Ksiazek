import numpy as np


def inverseMatrix(A):
    if type(A) is np.ndarray and np.any(A):
        if len(np.shape(A)) == 2 and np.linalg.det(A) != 0:
            if np.shape(A)[0] == np.shape(A)[1]:
                return np.linalg.inv(A)
        else:
            return None
    else:
        None


def addMatrices(A, B):
    if type(A) is np.ndarray and type(B) is np.ndarray and np.any(A) and np.any(B):
        if np.shape(A) == np.shape(B):
            return np.add(A, B)
        else:
            return None
    else:
        return None


def transposeMatrix(A):
    if type(A) is np.ndarray and np.any(A):
        return np.transpose(A)
    else:
        return None


def multiplySquareMatrices(A, B):
    if type(A) is np.ndarray and type(B) is np.ndarray and np.any(A) and np.any(B):
        if len(np.shape(A)) == 2 and len(np.shape(B)) == 2:
            if np.shape(A)[0] == np.shape(B)[1] and np.shape(A)[1] == np.shape(B)[0]:
                return np.dot(A, B)
    else:
        return None
