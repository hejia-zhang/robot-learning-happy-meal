import gym
import numpy as np


# Load the environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
QTable = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = .8
y = .95
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
total_reward_list = []
for i in range(num_episodes):
    # Rest environment and get first new observation
    observation = env.reset()
    it = 1
    total_reward = 0
    # The Q-Table learning algorithm
    while it <= 100:
        it += 1
        # Choose an action by greedily picking from Q table
        action = np.argmax(QTable[observation, :] + np.random.randn(1, env.action_space.n) * (1./(i + 1)))
        # Get new state and reward from environment
        observation_new, reward, done, _ = env.step(action)
        # Update Q-Table with new knowledge
        QTable[observation, action] = QTable[observation, action] + \
                                      lr * (reward + y * np.max(QTable[observation_new, :]) -
                                            QTable[observation, action])
        total_reward += reward
        observation = observation_new
        if done:
            break
    total_reward_list.append(total_reward)

print("Score over time: " + str(sum(total_reward_list)/num_episodes))
print("Final Q-Table Values")
print(QTable)



