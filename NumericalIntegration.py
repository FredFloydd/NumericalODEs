import numpy as np


def test_function(x, y):
    return x + 2 * y


def equation_5a_function(x, y):
    return -4 * y + 4 * np.exp(-2 * x)


def equation_8a_function(x, y):
    return 4 * y - 5 * np.exp(-x)


def euler_method(function, x_0, x_max, y_0, h):
    max_n = int((x_max - x_0) / h)
    y_values = np.zeros(max_n + 1)
    y_values[0] = y_0
    x = x_0
    for i in range(max_n):
        y_values[i + 1] = y_values[i] + h * function(x, y_values[i])
        x += h
    return y_values


def runge_kutta(function, x_0, x_max, y_0, h):
    max_n = int((x_max - x_0) / h)
    y_values = np.zeros(max_n + 1)
    y_values[0] = y_0
    x = x_0
    for i in range(max_n):
        k_1 = h * function(x, y_values[i])
        k_2 = h * function(x + h / 2.0, y_values[i] + k_1 / 2.0)
        k_3 = h * function(x + h / 2.0, y_values[i] + k_2 / 2.0)
        k_4 = h * function(x + h, y_values[i] + k_3)
        y_values[i + 1] = y_values[i] + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0
        x += h
    return y_values
