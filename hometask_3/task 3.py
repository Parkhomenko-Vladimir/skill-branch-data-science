import numpy as np


## посчитать значений производной функции $\cos(x) + 0.05x^3 + \log_2{x^2}$ в точке $x = 10$.
# определили функцию
def f(x):
    return np.cos(x) + 0.05 * np.power(x, 3) + np.log2(np.power(x, 2))

x=10
dx = 0.01

out = (f(x+dx) - f(x))/dx
# Ответ
print('task 1', out )




## посчитать значение градиента функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$ в точке $(10, 1)$
# определили функцию
def f(x1, x2):
    return x1**2 * np.cos(x2) + 0.05 * np.power(x2, 3) + 3 * np.power(x1, 3) * np.log2(x2**2)

x1=10
x2 = 1
dx = 0.01

# производная по х1
dirivativeX1 = (f(x1+dx,x2) - f(x1,x2))/dx
# производная по х2
dirivativeX2 = (f(x1,x2+dx) - f(x1,x2))/dx

# знаения градиента
grad = (dirivativeX1 , dirivativeX2)
print('task 2',grad)




## найти точку минимуму для функции $\cos(x) + 0.05x^3 + \log_2{x^2}$. Зафиксировать параметр $\epsilon = 0.001$, начальное значение принять равным 10.
def f(x):
    return np.cos(x) + 0.05 * np.power(x, 3) + np.log2(np.power(x, 2))
e = 0.0001
x = 10
dirivative =10

# если задать условия для производной меньше двух, то значение градиента будет перепрыгивать минимум
# на графике будет видно, что точка минимума очень "узкая"

while not -2 < dirivative < 2:
    dirivative = (f(x + e) - f(x)) / e
    x = x - dirivative * e

print('task 3', x)


import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = list(map(f, x))

plt.figure(figsize=(10,7))
plt.title('точка минимума в является узкой, поэтому значение градиента может ее перепрыгивать')
plt.plot(x, y)
plt.ylabel("Y")
plt.xlabel("X")
plt.show()


# найти точку минимуму для функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$. Зафиксировать параметр $\epsilon = 0.001$, начальные значения весов принять равным [4, 10].

def f(x1, x2):
    return x1**2 * np.cos(x2) + 0.05 * np.power(x2, 3) + 3 * np.power(x1, 3) * np.log2(x2**2)

e = 0.0001
x1 = 4
x2 = 10

dirivativeX1 =10
dirivativeX2 =10

while not (-1 < dirivativeX1 < 1 and -1 < dirivativeX2 < 1):
    dirivativeX1 = (f(x1 + e, x2) - f(x1, x2)) / e
    dirivativeX2 = (f(x1, x2 + e) - f(x1, x2)) / e

    x1 = x1 - dirivativeX1 * e
    x2 = x2 - dirivativeX2 * e

print('task 4 (x1,x2) = ' , (x1,x2))