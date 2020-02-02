#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from QLearning import QLearningTable
from time import *

from ns3gym import ns3env

env = gym.make('ns3-v0')
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n
print("ac_space: {}".format(ac_space))
print(", ac_space.n: {}".format(ac_space.n))

total_episodes = 200
max_env_steps = 100
env._max_episode_steps = max_env_steps

epsilon = 0.3               # exploration rate
epsilon_min = 0.001
epsilon_decay = 0.999

time_history = []
rew_history = []

train_time_history = []
predict_time_history = []

train_time_sum = []
predict_time_sum = []


ql = QLearningTable(actions=list(range(a_size)), e_greedy=epsilon)
# model = keras.Sequential()
# model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
# model.add(keras.layers.Dense(a_size, activation='softmax'))
# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


for e in range(total_episodes):

    state = env.reset()

    state = np.reshape(state, [1, s_size])
    state_n = np.argmax(state)
    print("state: ", state_n)
    rewardsum = 0
    for time_t in range(max_env_steps):

        # Choose action
        # if np.random.rand(1) < epsilon:
        #     action = np.random.randint(a_size)
        # else:
        #     action = np.argmax(model.predict(state)[0])
        begin_time = time()
        action = ql.choose_action(state_n)
        end_time = time()
        predict_time = end_time - begin_time
        print("程序运行时间： ", predict_time)
        # Step
        next_state, reward, done, _ = env.step(action)

        print("next_state: {}, action: {}, reward: {}, done: {}"
              .format(next_state, action, reward, done))

        if done:
            print("episode: {}/{}, time_t: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time_t, rewardsum, epsilon))
            break

        next_state = np.reshape(next_state, [1, s_size])
        next_state_n = np.argmax(next_state)
        print("next_state_n: ", next_state_n)

        # Train
        begin_time = time()
        ql.learn(r=reward, s_=next_state_n, done=done, s=state_n, a=action)
        end_time = time()
        train_time = end_time - begin_time
        print("程序训练时间： ", train_time)
        # target = reward
        # if not done:
        #     target = (reward + 0.95 * np.amax(model.predict(next_state)[0]))
        #
        # target_f = model.predict(state)
        # target_f[0][action] = target
        # model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state
        state_n = np.argmax(state)
        rewardsum += reward
        if epsilon > epsilon_min: epsilon *= epsilon_decay
        predict_time_sum.append(predict_time)
        train_time_sum.append(train_time)

        
    time_history.append(time_t)
    rew_history.append(rewardsum)

    predict_time_history.append(np.mean(predict_time_sum))
    train_time_history.append(np.mean(train_time_sum))

    predict_time_sum = []
    train_time_sum = []

#for n in range(2 ** s_size):
#    state = [n >> i & 1 for i in range(0, 2)]
#    state = np.reshape(state, [1, s_size])
#    print("state " + str(state) 
#        + " -> prediction " + str(model.predict(state)[0])
#        )

#print(model.get_config())
#print(model.to_json())
#print(model.get_weights())

print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")#, color='red')
plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")#, color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

np.save("./time_history.npy", time_history)
np.save("./reward.npy", rew_history)

plt.savefig('learning.pdf', bbox_inches='tight')
# plt.show()


print("Plot Learning Time Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Time Performance')
plt.plot(range(len(predict_time_history)), predict_time_history, label='Predict Time', marker="^", linestyle=":")#, color='red')
plt.plot(range(len(train_time_history)), train_time_history, label='Train Time', marker="", linestyle="-")#, color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

np.save("./ql_predict_time_history.npy", predict_time_history)
np.save("./ql_train_time_history.npy", train_time_history)

plt.savefig('ql_learning_time.pdf', bbox_inches='tight')
plt.show()
