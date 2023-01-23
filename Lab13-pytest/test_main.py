import pytest
import numpy as np
from main import *


def test1_inverseMatrix_results():
    test1 = np.array([[6, 1, 1],
                      [4, -2, 5],
                      [2, 8, 7]])
    got = inverseMatrix(test1)
    want = np.array([[0.17647059, -0.00326797, -0.02287582],
                     [0.05882353, -0.13071895, 0.08496732],
                     [-0.11764706, 0.1503268, 0.05228758]])

    assert np.allclose(got, want)


def test2_inverseMatrix_results2():
    test2 = np.array([[1, 0],
                      [0, 1]])
    got = inverseMatrix(test2)
    want = np.array([[1, 0],
                     [0, 1]])
    assert np.allclose(got, want)


def test3_inverseMatrix_wrong_input():
    test3 = 'Test'
    got = inverseMatrix(test3)

    assert got is None


def test4_inverseMatrix_wrong_dimension():
    test4 = np.array([1, 2, 3])
    got = inverseMatrix(test4)

    assert got is None


def test5_inverseMatrix_det_zero():
    test5 = np.array([[1, 1],
                      [1, 1]])
    got = inverseMatrix(test5)

    assert got is None


def test6_addMatrices_wrong_inputA():
    test6A = 'Test'
    test6B = np.array([1, 1])
    got = addMatrices(test6A, test6B)

    assert got is None


def test7_addMatrices_wrong_inputB():
    test7A = np.array([1, 1])
    test7B = 'Test'
    got = addMatrices(test7A, test7B)

    assert got is None


def test8_addMatrices_different_dimensions():
    test8A = np.array([1, 2])
    test8B = np.array([[3], [4]])
    got = addMatrices(test8A, test8B)

    assert got is None


def test9_addMatrices_results():
    test9A = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    test9B = np.array([[9, 8, 7],
                       [6, 5, 4],
                       [3, 2, 1]])
    got = addMatrices(test9A, test9B)
    want = np.array([[10, 10, 10],
                     [10, 10, 10],
                     [10, 10, 10]])

    assert np.array_equal(got, want)


def test10_addMatrices_emptyA():
    testA = np.array([])
    testB = np.array([[9, 8, 7],
                      [6, 5, 4],
                      [3, 2, 1]])
    got = addMatrices(testA, testB)

    assert got is None


def test11_addMatrices_emptyB():
    testA = np.array([[9, 8, 7],
                      [6, 5, 4],
                      [3, 2, 1]])
    testB = np.array([])
    got = addMatrices(testA, testB)

    assert got is None


def test12_transposeMatrix_input():
    testA = 'Test'
    got = transposeMatrix(testA)

    assert got is None


def test13_transposeMatrix_empty():
    testA = np.array([])
    got = transposeMatrix(testA)

    assert got is None


def test14_inverseMatrix_empty():
    testA = np.array([])
    got = inverseMatrix(testA)

    assert got is None


def test15_transposeMatrix_result1():
    testA = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    got = transposeMatrix(testA)
    want = np.array([[1, 4, 7],
                     [2, 5, 8],
                     [3, 6, 9]])

    assert np.array_equal(got, want)


def test16_transposeMatrix_result2():
    testA = np.array([[0.17647059, -0.00326797, -0.02287582],
                     [0.05882353, -0.13071895, 0.08496732],
                     [-0.11764706, 0.1503268, 0.05228758]])
    got = transposeMatrix(testA)
    want = np.array([[0.17647059, 0.05882353, -0.11764706],
                     [-0.00326797, -0.13071895, 0.1503268],
                     [-0.02287582, 0.08496732, 0.05228758]])

    assert np.array_equal(got, want)


def test17_transposeMatrix_result3():
    testA = np.array([1, 2, 3])
    got = transposeMatrix(testA)
    want = np.array([1, 2, 3])

    assert np.array_equal(got, want)


def test18_multiplyMatrices_emptyA():
    testA = np.array([])
    testB = np.array([[1, 2],
                      [3, 4]])
    got = multiplySquareMatrices(testA, testB)

    assert got is None


def test19_multiplyMatrices_emptyB():
    testA = np.array([[1, 2],
                      [3, 4]])
    testB = np.array([])
    got = multiplySquareMatrices(testA, testB)

    assert got is None

def test20_multiplyMatrices_imputA():
    testA = 'Test'
    testB = np.array([[1, 2],
                      [3, 4]])
    got = multiplySquareMatrices(testA, testB)

    assert got is None


def test21_multiplyMatrices_imputA():
    testA = np.array([[1, 2],
                      [3, 4]])
    testB = 'Test'
    got = multiplySquareMatrices(testA, testB)

    assert got is None


def test22_multiplyMatrices_result():
    testA = np.array([[1, 0],
                      [0, 1]])
    testB = np.array([[4, 1],
                      [2, 2]])
    got = multiplySquareMatrices(testA, testB)
    want = np.array([[4, 1],
                     [2, 2]])

    assert np.array_equal(got, want)


def test23_multiplyMatrices_dimension():
    testA = np.array([[1, 2],
                      [3, 4]])
    testB = np.array([1, 2])
    got = multiplySquareMatrices(testA, testB)

    assert got is None