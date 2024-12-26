import numpy as np

# System and Discretization Parameters
T = 1  # Sampling time

# Constraints
bound_x = np.array([[0, 10], [0, 10], [-np.pi, np.pi]])
bound_u = np.array([[0.25, 1], [-1, 1]])
bound_w = np.array([[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]])

# Abstraction parameters (reduced for faster computation)
n_x = np.array([100, 100, 30]) # State discretization
n_u = np.array([3, 5])  # Control input discretization

# Space discretization
d_x = (bound_x[:, 1] - bound_x[:, 0]) / n_x
p_x = np.concatenate(([1], np.cumprod(n_x)))
n_xi = np.prod(n_x)
grid_xi = np.arange(1, n_xi + 1).reshape(n_x)


# Control input discretization
d_u = (bound_u[:, 1] - bound_u[:, 0]) / (n_u - 1)
p_u = np.concatenate(([1], np.cumprod(n_u)))
n_sigma = np.prod(n_u)


# Noise discretization

d_w = (bound_w[:, 1] - bound_w[:, 0])


# System dynamics
def f(x, u, w):
    return np.array([
        x[0] + T * u[0] * np.cos(x[2]) + T * w[0],
        x[1] + T * u[0] * np.sin(x[2]) + T * w[1],
        x[2] + T * u[1] + T * w[2]
    ])


def Jf_x(u):
    return np.array([
        [1, 0, T * np.abs(u[0])],
        [0, 1, T * np.abs(u[0])],
        [0, 0, 1]
    ])

def Jf_w(u):
    return np.array([
        [T, 0, 0],
        [0, T, 0],
        [0, 0, T]
    ])


# Automaton transition relation
A_s = np.array([
    [1, 2, 3, 1, 5],
    [2, 2, 5, 4, 5],
    [3, 5, 3, 4, 5],
    [4, 4, 4, 4, 4],
    [5, 5, 5, 5, 5]
])

# Automaton parameters
n_s = len(A_s)
I_s = 1  # Initial state
F_s = [4]  # Final states

# Regions definition
R1 = np.array([[4, 5], [8.5, 9.5], [-np.pi, np.pi]])
R2 = np.array([[8.5, 9.5], [2, 3], [-np.pi, np.pi]])
R3 = np.array([[2, 3], [0.5, 1.5], [-np.pi, np.pi]])
R4 = np.array([[3, 7], [3, 7], [-np.pi, np.pi]])