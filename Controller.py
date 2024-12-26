import numpy as np
import time
from Utils import state2coord, h1, coordbox2state, labelStates
from Params import n_s, n_xi, n_sigma, F_s, grid_xi, p_x

class SymbolicController:

    def __init__(self, symbolicAbstraction):
        self.symbolicTransitions = symbolicAbstraction
        self.labelingXi = labelStates()

        # Choice of a sucessor sample 
        self.gSample = np.zeros((n_s, n_xi, n_sigma), dtype=int)
        for psi in range(n_s):
            self.gSample[psi, :, :] = symbolicAbstraction[:, :, 0]

        # Initialize value function and controller
        self.V = np.inf * np.ones((n_s, n_xi))
        self.V5 = np.zeros((n_s, n_xi), dtype=int)
        for f in F_s:
            self.V[f - 1, :] = 0
        
        self.h2 = np.zeros((n_s, n_xi), dtype=int)
    

    def Start(self, max_iter=100):
        self.h25 = np.zeros((n_s, n_xi), dtype=int)
        return self.Load().SynthesisController(max_iter)
    

    def SynthesisController(self, max_iter):
        # Fixed-point iteration
        l=time.time()
        for iter in range(max_iter):
            Vp = self.V.copy()
            
            for xi in range(n_xi):
                for psi in range(1, n_s + 1):
                    psiSucc = h1(psi, xi, self.labelingXi)

                    if self.V[psi - 1, xi] == np.inf:
                        if self.h2[psiSucc - 1, xi] != 0:
                            self.V5[psi - 1, xi] = iter 
                        else:
                            Vmax = np.zeros(n_sigma)
                            for sigma in range(n_sigma):
                                xi_succ = self.symbolicTransitions[xi, sigma, :]
                                if np.all(xi_succ):
                                    if Vp[psiSucc -1, self.gSample[psi - 1, xi, sigma] - 1] != np.inf:
                                        cMin = state2coord(int(xi_succ[0]), p_x)
                                        cMax = state2coord(int(xi_succ[1]), p_x)

                                        if cMin[2] <= cMax[2]: 
                                            qSucc = coordbox2state(cMin, cMax, grid_xi)
                                        else:
                                            qSucc = np.concatenate(
                                                (coordbox2state(cMin, [cMax[0], cMax[1], cMax[2]], grid_xi), 
                                                coordbox2state([cMin[0], cMin[1], 1], cMax, grid_xi)))
                                        
                                        vSucc = Vp[psiSucc - 1, qSucc - 1]
                                        Vmax[sigma] = np.all(vSucc != np.inf) 

                                        if Vmax[sigma]:
                                            break 
                                        else: 
                                            i_p = np.argmax(vSucc != np.inf)
                                            self.gSample[psi - 1, xi, sigma] = qSucc[i_p]
                        
                            if np.any(Vmax):
                                self.V5[psi - 1, xi] = iter 
                                sigma = np.max(Vmax)
                                self.h25[psiSucc - 1, xi] = sigma + 1

            
            # Check convergence
            if np.array_equal(Vp, self.V) and abs(time.time() - l) > 600: #timeout condition (divergence)
                print("Fixed-point algorithm reached convergence.")
                break
        
        controllerCount = np.count_nonzero(self.h2)
        totalStates = self.h2.size
        print(f"\nController Coverage: {controllerCount}/{totalStates} states")
        
        return self.V, self.h2
    

    #Saving and loading for time efficiency (avoiding recomputation)
    def Load(self):
        self.V = np.loadtxt('./Models/V_result.csv', delimiter=',')
        self.h2 = np.loadtxt('./Models/h2_result.csv', delimiter=',')
        return self

    def Save(self):
        np.savetxt('./Models/V_result.csv', self.V, delimiter=',')
        np.savetxt('./Models/h2_result.csv', self.h2, delimiter=',')