import numpy as np
import pandas as pd
import time

from Utils import state2coord, coord2state, ComputeControlDiscretization
from Params import bound_x, d_x, d_w, p_x, n_xi, n_sigma, bound_w

class SymbolicAbstraction:
    def __init__(self, f, Jf_x, Jf_w):
        self.f = f
        self.Jf_x = Jf_x
        self.Jf_w = Jf_w

        self.ControlDisc = ComputeControlDiscretization()


    def Start(self):
        self.g = self.computeSymbolicModel()
        return self 


    def computeSymbolicModel(self):
        print("Computing symbolic model...")
        
        # Disturbance at center
        wCenter = 0.5 * (bound_w[:, 0] + bound_w[:, 1])
        
        g = np.zeros((n_xi, n_sigma, 2), dtype=int)
        reachable_count = 0
        
        for xi in range(1, n_xi + 1):
            xiCenter = state2coord(xi, p_x)
            xCenter = bound_x[:, 0] + (xiCenter - 0.5) * d_x
            
            for sigma in range(1, n_sigma + 1):

                # Compute successor state
                xCenterSucc = self.f(xCenter, self.ControlDisc[:, sigma - 1], wCenter)

                # Compute deviation bounds (dxSucc) for the reachable set
                dxSucc = 0.5 * self.Jf_x(self.ControlDisc[:, sigma - 1]) @ d_x + 0.5 * self.Jf_w(self.ControlDisc[:, sigma - 1]) @ d_w
                
                reach = np.vstack([xCenterSucc - dxSucc, xCenterSucc + dxSucc])  # Reachable set

                if reach[0, 2] < -np.pi and reach[1, 2] >= -np.pi:
                    reach[0, 2] += 2 * np.pi
                elif reach[0, 2] < -np.pi and reach[1, 2] < -np.pi:
                    reach[0, 2] += 2 * np.pi
                    reach[1, 2] += 2 * np.pi
                elif reach[1, 2] > np.pi and reach[0, 2] <= np.pi:
                    reach[1, 2] -= 2 * np.pi
                elif reach[1, 2] > np.pi and reach[0, 2] > np.pi:
                    reach[1, 2] -= 2 * np.pi
                    reach[0, 2] -= 2 * np.pi

                reach = reach.transpose()
                if np.all(reach[:, 0] >= bound_x[:, 0]) and np.all(reach[:, 1] <= bound_x[:, 1]):
                    minSucc = coord2state(np.floor((reach[:, 0] - bound_x[:, 0]) / d_x).astype(int) + 1, p_x)
                    maxSucc = coord2state(np.ceil((reach[:, 1] - bound_x[:, 0]) / d_x).astype(int), p_x)
                    g[xi - 1, sigma - 1] = [minSucc, maxSucc]
                    
                    reachable_count += 1
        
        print(f"Reachable state-input pairks: {reachable_count}")
        return g
    

    def __getitem__(self, key):
        return self.g[key]


    #Saving and loading for time efficiency (avoiding recomputation)
    def Save(self, filename):
        rows = []
        
        for xi in range(n_xi):
            for sigma in range(n_sigma):

                minSucc = self.g[xi, sigma, 0]
                maxSucc = self.g[xi, sigma, 1]
                

                rows.append([xi + 1, sigma + 1, minSucc, maxSucc])
        
        df = pd.DataFrame(rows, columns=['State Index', 'Input Index', 'Min Successor', 'Max Successor'])
        
        df.to_csv("Models/"+filename, index=False)
        print(f"Symbolic model 'g' saved to {filename}.")


    def Load(self, filename="symbolic_model.csv"):
        df = pd.read_csv("Models/"+filename)
        
        g = np.zeros((n_xi, n_sigma, 2), dtype=int)
        
        for _, row in df.iterrows():
            xi = int(row['State Index']) - 1  
            sigma = int(row['Input Index']) - 1  
            minSucc = int(row['Min Successor'])
            maxSucc = int(row['Max Successor'])
            
            g[xi, sigma, 0] = minSucc
            g[xi, sigma, 1] = maxSucc
        
        print(f"Symbolic model 'g' loaded from '{filename}'")
        self.g = g

        return self
