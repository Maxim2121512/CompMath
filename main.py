import numpy as np


# B[0] - амплитуда, B[1] - частота, B[2] - фаза 
B = np.array([5.217, 94.893, 1.723])

# массив временных меток от start_time до end_time с частотой дискретизации sample_rate
start_time = 0
end_time = 2
sample_rate = 1000
times = np.arange(start_time, end_time, 1/sample_rate)

# lamda-функция для вычисления значений сигнала 
calc_signals = lambda B_i, x: B_i[0] * np.sin(2 * np.pi * B_i[1] * x + B_i[2])

# массив значений шумов
max_error = 1e-3
errors = np.random.normal(0, 1, len(times)) * max_error


# массив значений сигнала + шум
signal = calc_signals(B, times) + errors


# Дискретное преобразование Фурье
def dft(signal):
    N = len(signal)
 
    X = np.empty_like(signal, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * (k / N) * n)
    
    return X

# Поиск частоты сигнала
def find_freq(signal, sample_rate):
    spectrum = dft(signal)
    spectrum_magnitude = np.abs(spectrum)

    peak_idx = np.argmax(spectrum_magnitude)

    signal_frequency = peak_idx * (sample_rate / len(signal))
    
    return signal_frequency


# LUP-разложение, матрица C = L + U - E
def LUP(A):
    if (np.linalg.det(A) == 0):
        print("Матрица A - вырожденная")
        
    n = len(A)
    C = np.copy(A)
    P = np.eye(n)

    for i in range(n):
        pivot_val = 0.0
        pivot_row = -1

        for row in range(i, n):
            if (abs(C[row][i]) > pivot_val):
                pivot_val = abs(C[row][i])
                pivot_row = row
        
        if (pivot_val != 0):
            P[[i, pivot_row]] = P[[pivot_row, i]]
            C[[i, pivot_row]] = C[[pivot_row, i]]
        
            for j in range(i + 1, n):
                C[j][i] /= C[i][i]
                for k in range(i + 1, n):
                    C[j][k] -= C[j][i] * C[i][k]

    return C, P


# Решение Ly = Pb
def forward_substition(C, Pb):
    n = len(Pb)
    y = np.zeros(n)

    for i in range(0, n):
        y[i] = Pb[i]
        for j in range(0, i):
            y[i] -= C[i][j] * y[j]

    return y

# Решение Ux = y
def backward_substitution(C, y):
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= C[i][j] * x[j]
        x[i] /= C[i][i]

    return x

'''
МНК, используется для поиска первого приближения амплитуды и фазы
f(t,B) = A * sin(2pi*wt + p) = A * (sin(2pi * wt) * cos(p) + cos(2pi + wt) * sin(p))
X = (sin(2pi * wt), cos(2pi + wt))^T - матрица Nx2
y - вектор Nx1

m = A * cos(p)
k = A * sin(p) => k / m = tg(p) => p = arctg(k/m)

m^2 + k^2 = A^2 * (cos(p)^2 + sin(p)^2)
A = sqrt(m^2 + k^2)

Ищем оценку \Beta = (m, k)^T

(X^T * X)(оценка \Beta) = X^T * y
'''
def least_squares(y, X):
    A = X.T @ X
    b = X.T @ y
    C, P = LUP(A)
    y = forward_substition(C=C, Pb=(P @ b))
    x = backward_substitution(C=C, y=y)

    amplitude = np.sqrt(x[0]**2 + x[1]**2)
    phase = np.arctan(x[1] / x[0])

    return amplitude, phase



# Вычисление Якобиана
def calculate_J(B_i, times):
    A = np.zeros((len(times), 3))
    dr_dA = -np.sin(2 * np.pi * B_i[1] * times + B_i[2])
    dr_dw = -2 * np.pi * B_i[0] * times * np.cos(2 * np.pi * B_i[1] * times + B_i[2])
    dr_dp = -B_i[0] * np.cos(2 * np.pi * B_i[1] * times + B_i[2])

    for i in range(len(A)):
        A[i] = dr_dA[i], dr_dw[i], dr_dp[i]

    return A 

'''
Решение системы Ax = b, где 
A = J^T * W * J, 
b = -J^T * W * r(B^i), 
x = (B^(i+1) - B^i)^T  
'''
def solve_system(B_i, J, W, r):
    temp = J.T @ W
    A = temp @ J
    b = -(temp @ r)
    C, P = LUP(A)
    y = forward_substition(C=C, Pb=(P @ b))
    x = backward_substitution(C=C, y=y)
    return B_i + x

# Метод Гаусса-Ньютона
def Gauss_Newton(B_i, x, y, eps):

    error_variance = np.var(eps)
    W = np.diag(np.repeat(1/error_variance, len(eps)))
    J = calculate_J(B_i, x)
    r = y - calc_signals(B_i, x)
    
    return solve_system(B_i, J, W, r), r



print("Начальные данные:\nЧастота: {}\nАмлитуда: {}\nФаза: {}\n".format(B[1], B[0], B[2]))

freq = find_freq(signal=signal, sample_rate=sample_rate)
amplitude, phase = least_squares(y=signal, X=np.array([np.sin(2 * np.pi * freq * times), np.cos(2 * np.pi * freq * times)]).T)

print("Первое приближение:\nЧастота: {}\nАмплитуда: {}\nФаза: {}\n".format(freq, amplitude, phase))

B[0], B[1], B[2] = amplitude, freq, phase

for i in range(0, 10):
    B, r = Gauss_Newton(B_i=B, x=times, y=signal, eps=errors)
    print("Шаг: {}\nСумма квадратов остатков: {}\nЧастота: {}\nАмплитуда: {}\nФаза: {}\n".format(i, np.sum(r ** 2), B[1], B[0], B[2]))