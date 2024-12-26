import numpy as np
from Params import bound_x, bound_u, d_x, p_x, n_xi, n_sigma, d_u, p_u, A_s, R1, R2, R3, R4

def state2coord(state, p_x):
    I = np.zeros(len(p_x)-1, dtype=int)
    for k in range(len(p_x) - 1):
        mod = (state - 1) % p_x[k + 1]
        I[k] = np.floor(mod / p_x[k]) + 1

    return I

def coord2state(I, p_x):
    return (I - 1).transpose() @ p_x[0: len(I)] + 1

def q(x):
    return coord2state(np.floor((x - bound_x[:, 0]) / d_x).astype(int) + 1, p_x)

def Lab_x(x):
    if np.all(x >= R1[:, 0]) and np.all(x <= R1[:, 1]):
        return 2
    elif np.all(x >= R2[:, 0]) and np.all(x <= R2[:, 1]):
        return 3
    elif np.all(x >= R3[:, 0]) and np.all(x <= R3[:, 1]):
        return 4
    elif np.all(x >= R4[:, 0]) and np.all(x <= R4[:, 1]):
        return 5
    return 1 

def labelStates():
    labelingXi = np.zeros(n_xi, dtype=int)
    for xi in range(n_xi):
        c_xi = state2coord(xi + 1, p_x)
        x_c = bound_x[:, 0] + (c_xi - 0.5) * d_x
        labelingXi[xi] = Lab_x(x_c)
    return labelingXi


def coordbox2state(Imin, Imax, box):
    k = [range(Imin[i], min(Imax[i] + 1, box.shape[i])) for i in range(len(Imin))]
    L = box[np.ix_(*k)]
    L = L.reshape(-1, 1)
    return L


def ComputeControlDiscretization():
    disc = np.zeros((len(bound_u), n_sigma))
    for i in range(1, n_sigma + 1):
        disc[:, i - 1] = bound_u[:, 0] + (state2coord(i, p_u) - 1) * d_u
    
    return disc


def h1(psi, xi, labelingXi):
    return A_s[psi - 1, labelingXi[xi] - 1]
    