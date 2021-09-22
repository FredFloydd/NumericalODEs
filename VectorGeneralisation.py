import numpy as np


def equation_16_z(x, y, z, p, alpha):
    return - p * p * np.power(1 + x, -alpha) * y


def equation_16_y(x, y, z, p, alpha):
    return z


def equation_14_solution(x, p):
    return - (1 + x) * np.sin(-p + p / (1 + x)) / p


def vector_runge_kutta(z_function, y_function, x_0, x_max, y_0, z_0, h, p, alpha):
    max_n = int((x_max - x_0) / h)
    y_values = np.zeros(max_n + 1)
    z_values = np.zeros(max_n + 1)
    y = y_0
    y_values[0] = y_0
    z = z_0
    z_values[0] = z_0
    x = x_0
    for i in range(max_n):
        k_z_1 = h * z_function(x, y, z, p, alpha)
        k_y_1 = h * y_function(x, y, z, p, alpha)
        k_z_2 = h * z_function(x + h / 2.0, y + k_y_1 / 2.0, z + k_z_1 / 2.0, p, alpha)
        k_y_2 = h * y_function(x + h / 2.0, y + k_y_1 / 2.0, z + k_z_1 / 2.0, p, alpha)
        k_z_3 = h * z_function(x + h / 2.0, y + k_y_2 / 2.0, z + k_z_2 / 2.0, p, alpha)
        k_y_3 = h * y_function(x + h / 2.0, y + k_y_2 / 2.0, z + k_z_2 / 2.0, p, alpha)
        k_z_4 = h * z_function(x + h, y + k_y_3, z + k_z_3, p, alpha)
        k_y_4 = h * y_function(x + h, y + k_y_3, z + k_z_3, p, alpha)
        z = z_values[i] + (k_z_1 + 2 * k_z_2 + 2 * k_z_3 + k_z_4) / 6.0
        y = y_values[i] + (k_y_1 + 2 * k_y_2 + 2 * k_y_3 + k_y_4) / 6.0
        z_values[i + 1] = z
        y_values[i + 1] = y
        x += h
    return y_values


def calculate_new_p(z_function, y_function, x_0, x_max, y_0, z_0, h, p_1, p_2, alpha):
    g_p_1 = vector_runge_kutta(z_function, y_function, x_0, x_max, y_0, z_0, h, p_1, alpha)[-1]
    g_p_2 = vector_runge_kutta(z_function, y_function, x_0, x_max, y_0, z_0, h, p_2, alpha)[-1]
    return (g_p_2 * p_1 - g_p_1 * p_2) / (g_p_2 - g_p_1)


def false_position(z_function, y_function, x_0, x_max, y_0, z_0, h, p_1, p_2, alpha, epsilon):
    finished = False
    count = 1
    while not finished:
        p_s = calculate_new_p(z_function, y_function, x_0, x_max, y_0, z_0, h, p_1, p_2, alpha)
        g_p_s = vector_runge_kutta(z_function, y_function, x_0, x_max, y_0, z_0, h, p_s, alpha)[-1]
        count += 1
        if np.absolute(g_p_s) < epsilon:
            return p_s
        else:
            if p_s * p_1 < 0:
                p_2 = p_s
            else:
                p_1 = p_s


def find_eigenvalues(z_function, y_function, x_0, x_max, y_0, z_0, h, alpha, n, tolerance, step):
    p_1 = 0
    p_2 = step
    eigenvalues = np.zeros(n)
    for i in range(n):
        g_p_1 = vector_runge_kutta(z_function, y_function, x_0, x_max, y_0, z_0, h, p_1, alpha)[-1]
        g_p_2 = vector_runge_kutta(z_function, y_function, x_0, x_max, y_0, z_0, h, p_2, alpha)[-1]
        while g_p_1 * g_p_2 > 0:
            p_1 = p_2
            g_p_1 = g_p_2
            p_2 += step
            g_p_2 = vector_runge_kutta(z_function, y_function, x_0, x_max, y_0, z_0, h, p_2, alpha)[-1]
        epsilon = np.absolute(tolerance * (g_p_2 - g_p_1) / (10 * step))
        p_i = false_position(z_function, y_function, x_0, x_max, y_0, z_0, h, p_1, p_2, alpha, epsilon)
        eigenvalues[i] = p_i
        p_1 = p_i + tolerance
        p_2 = p_i + 2 * tolerance
    return eigenvalues


def normalise(y_values, x_0, x_max, p, h):
    x_values = np.arange(x_0, x_max + h/2, h)
    y_values = np.array(y_values)
    integrand = p * p * y_values * y_values * np.power(1 + x_values, -8)
    integral = np.trapz(integrand, dx=h)
    coefficient = np.power(integral, 0.5)
    y_values = y_values / coefficient
    return y_values


def generate_eigenfunctions(eigenvalues, x_0, x_max, y_0, z_0, h, alpha):
    functions = []
    for i in range(eigenvalues.shape[0]):
        function = vector_runge_kutta(equation_16_z, equation_16_y, x_0, x_max, y_0, z_0, h, eigenvalues[i], alpha)
        function = normalise(function, x_0, x_max, eigenvalues[i], h)
        functions.append(function)
    functions = np.array(functions)
    return functions
