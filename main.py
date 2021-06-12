import numpy as np
import matplotlib.pyplot as plt


def newton(x_init, f, f_g, f_h, epsilon=1e-10, max_iterations=10):
    x = x_init
    step_sizes = []
    pos = [np.copy(x_init)]
    print(f"starting newton method from x0={x_init}")
    for i in range(max_iterations):
        gradient = f_g(x)
        transposed_gradient = np.transpose(gradient)
        hessian = f_h(x)
        inverse_hessian = np.linalg.inv(hessian)

        direction = -inverse_hessian @ gradient
        decrement = transposed_gradient @ inverse_hessian @ gradient

        if decrement / 2 <= epsilon:
            return x, i + 1, step_sizes, pos

        step_size = iterate_step_length(x, f, direction, gradient)
        x += step_size * direction
        step_sizes.append(step_size)
        pos.append(np.copy(x))
        print(f'{i + 1}) x={x}, direction={direction}, stepsize={step_size}')

    return x, max_iterations, step_sizes, pos


def steepest_descent(x_init, f, f_g, f_h, epsilon=1e-6, max_iterations=10000):
    x = x_init
    step_sizes = []
    pos = [np.copy(x_init)]
    print(f"starting newton method from x0={x_init}")
    for i in range(max_iterations):
        gradient = f_g(x)
        direction = - gradient
        if np.linalg.norm(gradient) <= epsilon:
            return x, i + 1, step_sizes, pos

        step_size = iterate_step_length(x, f, direction, gradient)
        x += step_size * direction
        step_sizes.append(step_size)
        pos.append(np.copy(x))
        print(f'{i + 1}) x={x}, direction={direction}, stepsize={step_size}')

    return x, max_iterations, step_sizes, pos


def iterate_step_length(x, f, direction, gradient, rho=0.49, beta=0.99):
    step_size = 1
    while f(x + step_size * direction) > f(x) + rho * step_size * np.inner(gradient, direction):
        step_size *= beta
    return step_size


def plot_step_size(f, step_sizes, pos, minimizer, title, min_x=-2, max_x=2, min_y=-2, max_y=2, num=250):
    plt.figure(figsize=(16, 9))
    plt.plot(range(len(step_sizes)), step_sizes)
    plt.title(f'{title} - step size')
    plt.xlabel('iterations')
    plt.ylabel('step-size')
    plt.show()

    x = np.linspace(min_x, max_x, num)
    y = np.linspace(min_y, max_y, num)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))
    plt.figure(figsize=(16, 9))
    plt.contour(X, Y, Z, 50, cmap='jet')

    x1 = [x[0] for x in pos]
    x2 = [x[1] for x in pos]
    plt.plot(x1, x2, color='b', alpha=0.25)
    plt.scatter(x1, x2, color='y', alpha=0.5, label='steps')
    plt.scatter(minimizer[0], minimizer[1], color='r', marker='o', label='minimizer')
    plt.scatter(pos[0][0], pos[0][1], color='b', marker='*', label='start-point')
    plt.scatter(pos[-1][0], pos[-1][1], color='g', marker='*', label='my-solution')
    plt.title(f'{title} / my-solution:{pos[-1]}')
    plt.legend()
    plt.show()


def sphere(x):
    return (x[0] - 1) ** 4 + (x[1] - 1) ** 4


def sphere_grad(x):
    return np.array([
        4 * (x[0] - 1) ** 3,
        4 * (x[1] - 1) ** 3
    ])


def sphere_hess(x):
    return np.array([
        [12 * (x[0] - 1) ** 2, 0],
        [0, 12 * (x[1] - 1) ** 2]
    ])


def matyas(x):
    return 0.26 * (x[0] ** 4 + x[1] ** 4) - 0.48 * x[0] * x[1]


def matyas_grad(x):
    return np.array([
        1.04 * x[0] ** 3 - 0.48 * x[1],
        1.04 * x[1] ** 3 - 0.48 * x[0]
    ])


def matyas_hess(x):
    return np.array([
        [3.12 * x[0] ** 2, -0.48],
        [-0.48, 3.12 * x[1] ** 2]
    ])


def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 4 + (2 * x[0] + x[1] - 5) ** 4


def booth_grad(x):
    return np.array([
        8 * (2 * x[0] + x[1] - 5) ** 3 + 4 * (x[0] + 2 * x[1] - 7) ** 3,
        4 * (2 * x[0] + x[1] - 5) ** 3 + 8 * (x[0] + 2 * x[1] - 7) ** 3
    ])


def booth_hess(x):
    return np.array([
        [12 * (4 * (2 * x[0] + x[1] - 5) ** 2 + (x[0] + 2 * x[1] - 7) ** 2),
         24 * ((2 * x[0] + x[1] - 5) ** 2 + (x[0] + 2 * x[1] - 7) ** 2)],
        [24 * ((2 * x[0] + x[1] - 5) ** 2 + (x[0] + 2 * x[1] - 7) ** 2),
         12 * ((2 * x[0] + x[1] - 5) ** 2 + 4 * (x[0] + 2 * x[1] - 7) ** 2)]
    ])


def himmel(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmel_grad(x):
    return np.array([
        (4 * x[0] ** 3 + 4 * x[0] * x[1] - 42 * x[0] + 2 * x[1] ** 2 - 14),
        (4 * x[1] ** 3 + 4 * x[0] * x[1] - 26 * x[1] + 2 * x[0] ** 2 - 22)
    ])


def himmel_hess(x):
    return np.array([
        [4 * (x[0] ** 2 + x[1] - 11) + 8 * x[0] ** 2 + 2, 4 * x[0] + 4 * x[1]],
        [4 * x[0] + 4 * x[1], 4 * (x[0] + x[1] ** 2 - 7) + 8 * x[1] ** 2 + 2]
    ])


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


def qof_0(x):
    A, b = load_example(0)
    return qof(x, A, b)


def qof_0_grad(x):
    A, b = load_example(0)
    return qof_grad(x, A, b)


def qof_0_hess(x):
    A, b = load_example(0)
    return qof_hess(x, A, b)


def check_convergence(pos, grad):
    print('Check convergence\n------------')
    error = 1e-3
    for p in pos:
        gv = grad(p)
        if all([-error <= g <= error for g in gv]):
            print([0, 0])
        else:
            print(gv)


p = np.array([0.0, 0.0])
x_min, it, step_sizes, pos = steepest_descent(p, qof_0, qof_0_grad, qof_0_hess)
check_convergence(pos, qof_0_grad)

''' steepest descent
p = np.array([-1.5, 1.5])
minimizer = np.array([1, 1])
x_min, it, step_sizes, pos = steepest_descent(p, sphere, sphere_grad, sphere_hess)
plot_step_size(sphere, step_sizes, pos, minimizer, 'steepest_descent', -2, 4, -2, 4)

p = np.array([-1.9, 1.5])
minimizer = np.array([0.679366, 0.679366])
x_min, it, step_sizes, pos = steepest_descent(p, matyas, matyas_grad, matyas_hess)
plot_step_size(matyas, step_sizes, pos, minimizer, 'steepest_descent')

p = np.array([-1.0, 0.0])
minimizer = np.array([1, 3])
x_min, it, step_sizes, pos = steepest_descent(p, booth, booth_grad, booth_hess)
plot_step_size(booth, step_sizes, pos, minimizer, 'steepest_descent', -2, 5, -1, 6)

p = np.array([4.0, 4.0])
minimizer = np.array([3, 2])
x_min, it, step_sizes, pos = steepest_descent(p, himmel, himmel_grad, himmel_hess)
plot_step_size(himmel, step_sizes, pos, minimizer, 'steepest_descent', -5, 5, -5, 5)
'''

''' old try0
p = np.array([-1.0, 0.5])
minimizer = np.array([-0.679366, -0.679366])
x_min, it, step_sizes, pos = newton(p, matyas, matyas_grad, matyas_hess)
plot_step_size(matyas, step_sizes, pos, minimizer, f'newton')

p = np.array([-1.5, 1.5])
minimizer = np.array([1, 1])
x_min, it, step_sizes, pos = newton(p, f_0, f_0_g, f_0_h)
plot_step_size(f_0, step_sizes, pos, minimizer, 'newton')

'''
