import numpy as np
import matplotlib.pyplot as plt


def newton(x_init, f, f_g, f_h, epsilon=1e-10, max_iterations=100):
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
        # print(f'{i + 1}) x={x}, direction={direction}, stepsize={step_size}')

    return x, max_iterations, step_sizes, pos


def steepest_descent(x_init, f, f_g, f_h, epsilon=1e-3, max_iterations=10000):
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
        # print(f'{i + 1}) x={x}, direction={direction}, stepsize={step_size}')

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


def f_0(x):
    return (x[0] - 1) ** 4 + (x[1] - 1) ** 4


def f_0_g(x):
    return np.array([
        4 * (x[0] - 1) ** 3,
        4 * (x[1] - 1) ** 3
    ])


def f_0_h(x):
    return np.array([
        [12 * (x[0] - 1) ** 2, 0],
        [0, 12 * (x[1] - 1) ** 2]
    ])

def f_1(x):
    return x[0]**3+x[1]**2


def f_1_g(x):
    return np.array([
        3*x[0]**2,
        2*x[1]
    ])


def f_1_h(x):
    return np.array([
        [6*x[0], 0],
        [0, 2]
    ])

'''
p = np.array([1, 0.5])
minimizer = np.array([0, 0])
x_min, it, step_sizes, pos = steepest_descent(p, f_1, f_1_g, f_1_h)
plot_step_size(f_1, step_sizes, pos, minimizer, f'newton - {p}')

'''

p = np.array([-1.5, 1.5])
minimizer = np.array([1, 1])
x_min, it, step_sizes, pos = newton(p, f_0, f_0_g, f_0_h)
plot_step_size(f_0, step_sizes, pos, minimizer, 'newton')

p = np.array([-1.5, 1.5])
minimizer = np.array([1, 1])
x_min, it, step_sizes, pos = steepest_descent(p, f_0, f_0_g, f_0_h)
plot_step_size(f_0, step_sizes, pos, minimizer, 'steepest_descent')
