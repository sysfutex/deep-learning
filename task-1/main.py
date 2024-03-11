import numpy as np
import math as m
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# Размер выборки
N = 20

# Предел ошибки
eps0 = 0.1

# Коэффициенты полинома
a = np.random.uniform(-3, 3)
b = np.random.uniform(-3, 3)
c = np.random.uniform(-3, 3)
d = np.random.uniform(-3, 3)


def f1(x):
    return a * x ** 3 + b * x ** 2 + c * x + d


def f2(x):
    return x * m.sin(2 * m.pi * x)


# Аргументы выборки
x = np.random.uniform(-1, 1, N)

# Значения выборки (полином, равномерное распределение ошибки)
ypu = []
for i in range(N):
    eps = np.random.uniform(-eps0, eps0)
    ypu.append(f1(x[i]) + eps)

# Значения выборки (полином, нормальное распределение ошибки)
ypn = []
for i in range(N):
    eps = np.random.normal(0, eps0)
    while eps < -eps0 or eps > eps0:
        eps = np.random.normal(0, eps0)

    ypn.append(f1(x[i]) + eps)

# Значения выборки (x * sin(2 * pi * x), равномерное распределение ошибки)
ysu = []
for i in range(N):
    eps = np.random.uniform(-eps0, eps0)
    ysu.append(f2(x[i]) + eps)

# Значения выборки (x * sin(2 * pi * x), нормальное распределение ошибки)
ysn = []
for i in range(N):
    eps = np.random.normal(0, eps0)
    while eps < -eps0 or eps > eps0:
        eps = np.random.normal(0, eps0)

    ysn.append(f2(x[i]) + eps)


def create_null_matrix(h, w):
    matrix = []
    for i in range(h):
        row = []
        for j in range(w):
            row.append(0)
        matrix.append(row)

    return matrix


# Нахождение коэффициентов для полиномиальной регрессии (стр. 64)
def find_regression_coefficients(x, y, deg):
    def get_coefficient_matrix():
        matrix = create_null_matrix(deg + 1, deg + 1)

        for i in range(deg + 1):
            for j in range(deg + 1):
                for k in range(N):
                    matrix[i][j] += x[k] ** (i + j)

        return matrix

    def get_constant_matrix():
        matrix = create_null_matrix(deg + 1, 1)

        for i in range(deg + 1):
            for k in range(N):
                matrix[i][0] += (x[k] ** i) * y[k]

        return matrix

    return linalg.solve(get_coefficient_matrix(), get_constant_matrix())


# Количество точек функции
points = 100

# Оригинальные аргументы
original_x = np.linspace(-1.5, 1.5, points)

# Оригинальные значения (полином)
original_yp = []
for i in range(points):
    original_yp.append(f1(original_x[i]))

# Оригинальные значения (x * sin(2 * pi * x))
original_ys = []
for i in range(points):
    original_ys.append(f2(original_x[i]))

# Степень полинома для полиномиальной регрессии
M = 6

# Нахождения многочлена для полиномиальной регрессии (полином, равномерное распределение ошибки)
regression_coefficients_pu = find_regression_coefficients(x, ypu, M)
regression_polynomial_pu = []
for i in range(points):
    val = 0

    # w0 + w1 * x + w2 * x^2 + ... + wM * x^M
    for k in range(M + 1):
        val += regression_coefficients_pu[k] * original_x[i] ** k

    regression_polynomial_pu.append(val)

# Нахождение многочлена для полиномиальной регрессии (полином, нормальное распределение ошибки)
regression_coefficients_pn = find_regression_coefficients(x, ypn, M)
regression_polynomial_pn = []
for i in range(points):
    val = 0

    # w0 + w1 * x + w2 * x^2 + ... + wM * x^M
    for k in range(M + 1):
        val += regression_coefficients_pn[k] * original_x[i] ** k

    regression_polynomial_pn.append(val)

# Нахождение многочлена для полиномиальной регрессии (x * sin(2 * pi * x), равномерное распределение ошибки)
regression_coefficients_su = find_regression_coefficients(x, ysu, M)
regression_polynomial_su = []
for i in range(points):
    val = 0

    # w0 + w1 * x + w2 * x^2 + ... + wM * x^M
    for k in range(M + 1):
        val += regression_coefficients_su[k] * original_x[i] ** k

    regression_polynomial_su.append(val)

# Нахождение многочлена для полиномиальной регрессии (x * sin(2 * pi * x), нормальное распределение ошибки)
regression_coefficients_sn = find_regression_coefficients(x, ysn, M)
regression_polynomial_sn = []
for i in range(points):
    val = 0

    # w0 + w1 * x + w2 * x^2 + ... + wM * x^M
    for k in range(M + 1):
        val += regression_coefficients_sn[k] * original_x[i] ** k

    regression_polynomial_sn.append(val)

# График (полином, равномерное распределение ошибки)
plt.clf()
plt.plot(original_x, original_yp, color="blue")  # Оригинальный график
plt.plot(original_x, regression_polynomial_pu, color="green")  # Полиномиальная регрессия
plt.scatter(x, ypu, color="red")  # Выборка
plt.ylim(-5, 5)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./out/pu")

# График (полином, нормальное распределение ошибки)
plt.clf()
plt.plot(original_x, original_yp, color="blue")  # Оригинальный график
plt.plot(original_x, regression_polynomial_pn, color="green")  # Полиномиальная регрессия
plt.scatter(x, ypn, color="red")  # Выборка
plt.ylim(-5, 5)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./out/pn")

# График (x * sin(2 * pi * x), равномерное распределение ошибки)
plt.clf()
plt.plot(original_x, original_ys, color="blue")  # Оригинальный график
plt.plot(original_x, regression_polynomial_su, color="green")  # Полиномиальная регрессия
plt.scatter(x, ysu, color="red")  # Выборка
plt.ylim(-5, 5)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./out/su")

# График (x * sin(2 * pi * x), нормальное распределение ошибки)
plt.clf()
plt.plot(original_x, original_ys, color="blue")  # Оригинальный график
plt.plot(original_x, regression_polynomial_sn, color="green")  # Полиномиальная регрессия
plt.scatter(x, ysn, color="red")  # Выборка
plt.ylim(-5, 5)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./out/sn")
