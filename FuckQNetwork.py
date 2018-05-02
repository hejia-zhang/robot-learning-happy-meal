import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
QOut = tf.matmul(inputs1, W)
predict = tf.argmax(QOut, 1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - QOut))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000

# create lists to contain total rewards and steps per episode
step_list = []
total_reward_list = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Rest environment and get first new observation
        observation = env.reset()
        it = 1
        total_reward = 0
        # The Q-Table learning algorithm
        while it <= 100:
            it += 1
            # Choose an action by greedily picking from Q table
            action, QOutList = sess.run([predict, QOut],
                                        feed_dict={inputs1: np.identity(16)[observation: observation+1]})
            if np.random.rand(1) < e:
                action[0] = env.action_space.sample()
            # Get new state and reward from environment
            observation_new, reward, done, _ = env.step(action[0])
            # Obtain the Q' values by feeding the new state through our network
            QNewOutList = sess.run(QOut, feed_dict={inputs1: np.identity(16)[observation_new: observation_new+1]})
            # Obtain maxQ' and set our target value for chosen action
            maxQNew = np.max(QNewOutList)
            targetQ = QOutList
            targetQ[0, action[0]] = reward + y * maxQNew
            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[observation: observation+1],
                                                          nextQ: targetQ})
            total_reward += reward
            observation = observation_new
            if done:
                # Reduce chance of random action as we train the model
                e = 1./((i / 50) + 10)
                break
        total_reward_list.append(total_reward)
        step_list.append(it)

print("Percent of succesful episodes: " + str(sum(total_reward_list)/num_episodes) + "%")




