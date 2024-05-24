import numpy as np

iterations = 1000
# Define the grid size
grid_shape = (4, 4)
num_states = np.prod(grid_shape)

# Define the rewards and transition probabilities
reward = -1  # Reward for all transitions
actions = ["up", "down", "left", "right"]
num_actions = len(actions)


# Define the state transition function
def step(state, action):
    x, y = divmod(state, grid_shape[1])

    if action == "up":
        x = max(0, x - 1)
    elif action == "down":
        x = min(grid_shape[0] - 1, x + 1)
    elif action == "left":
        y = max(0, y - 1)
    elif action == "right":
        y = min(grid_shape[1] - 1, y + 1)

    new_state = x * grid_shape[1] + y
    return new_state, reward


# Initialize the value function for all states
V = np.zeros(num_states)

# Define the policy - uniform random policy
policy = np.ones([num_states, num_actions]) / num_actions

# Discount factor
gamma = 1  # Assuming no discounting as it's not specified

# Value iteration algorithm
for i in range(iterations):
    delta = 0
    values = np.copy(V)
    for state in range(num_states):
        # Skip terminal states (shaded states)
        if state in [0, 15]:
            continue

        v = 0
        for action in range(num_actions):
            next_state, reward = step(state, actions[action])
            v += policy[state, action] * (reward + gamma * V[next_state])
        values[state] = v * gamma
    # State value function update
    V = values

# Reshape the value function to match the grid shape for display
V_grid = V.reshape(grid_shape)
print(f"The value function for each state in the grid is ({iterations} iterations):")
print(V_grid)