import numpy as np


def qof(x, A, b):
    return (1 / 2) * x.T @ A @ x - b @ x


def qof_grad(x, A, b):
    return A @ x - b


def qof_hess(x, A, b):
    return A


def generate_pos_sem(n):
    A = np.random.randint(0, 3, size=(n, n))
    B = np.dot(A, A.T)
    return B


def get_matrix_path(i):
    return f'examples/matrix{i}.csv'


def generate_test_data(num, dim):
    for i in range(num):
        with open(get_matrix_path(i), 'w') as f:
            np.savetxt(f, generate_pos_sem(dim), delimiter=',', fmt='%d')


def load_example(i):
    A = np.loadtxt(get_matrix_path(i), delimiter=',')
    A = A.astype('float64')
    return A, np.ones(A.shape[0])

# generate_test_data(5, 12)