import numpy as np

# Given parameters
Reward = np.array([-2, -2, -2, 10, -1, 0, -1])

# Correct the transition probabilities matrix to ensure all rows sum to 1
Population = np.array([[0, .5, 0, 0, 0, .5, 0],
                       [0, 0, .8, 0, 0, 0, -2],
                       [0, 0, 0, .6, .4, 0, 0],
                       [0, 0, 0, 0, 0, 0, .1],
                       [.2, .4, .4, 0, 0, 0, 0],
                       [.1, 0, 0, 0, 0, .9, 0],
                       [0, 0, 0, 0, 0, 0, 0]])

gamma = 0.9

# Initialize value function to ones
V = np.ones(len(Reward))

# Convergence threshold
threshold = 0.001
delta = float('inf')

# Iteration counter
iteration = 0

while delta > threshold:
    iteration += 1
    V_prev = np.copy(V)
    V = np.dot(Population, (gamma * V)) + Reward
    delta = np.max(np.abs(V - V_prev))

print(f"Converged in {iteration} iterations.")
print("Estimated optimal value for each state (Iterative):")
print(V)

# ---------------------------------------------------------------------------

# Direct solution
I = np.eye(len(Population))
I_gamma_P = I - gamma * Population
I_gamma_P_inv = np.linalg.inv(I_gamma_P)
ss = np.dot(I_gamma_P_inv, Reward)

print("-----------------------------------------------")
print("Estimated optimal value for each state (Direct):")
print(ss)


