# Symbolic Control for Nonlinear Systems

This project implements symbolic abstraction and control for nonlinear systems using a structured methodology. It includes the computation of symbolic models, synthesis of controllers, and evaluation of the system's performance through simulations.

## Features

- Symbolic abstraction of nonlinear systems.
- Synthesis of symbolic controllers for safety, reachability, and other specifications.
- Integration with system dynamics and parameters.

## Project Structure
### Files and Their Purpose

- **`Abstraction.py`**
  - Implements the symbolic abstraction process.
  - Computes the symbolic model using discretized states and inputs.

- **`Controller.py`**
  - Synthesizes controllers for the symbolic system using fixed-point iteration.
  - Optimizes controllers based on specified objectives.

- **`Params.py`**
  - Defines system dynamics and parameters:
    - Constraints for states, control inputs, and disturbances.
    - Discretization settings for states and control inputs.
  - Provides the main dynamics (`f`) and Jacobians for the system.

- **`Utils.py`**
  - Utility functions for coordinate transformations, state labeling, and value computations.


## Installation: 

1. Clone the repository:
   ```bash
   git clone https://github.com/elEsquina/Symbolic-Control-For-Non-Linear-Systems.git
   cd Symbolic-Control-For-Non-Linear-Systems
   ```

2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```