import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ql_predict_time_history = np.load('./ql_predict_time_history.npy')
ql_train_time_history = np.load('./ql_train_time_history.npy')

dqn_predict_time_history = np.load('./dqn_predict_time_history.npy')
dqn_train_time_history = np.load('./dqn_train_time_history.npy')


print("Plot Learning Time Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Time Performance')
plt.plot(range(len(ql_predict_time_history)), ql_predict_time_history, label='QL Predict Time', marker="", linestyle=":", color='red')
plt.plot(range(len(ql_train_time_history)), ql_train_time_history, label='QL Train Time', marker="", linestyle="-", color='k')

plt.plot(range(len(dqn_predict_time_history)), dqn_predict_time_history, label='DQN Predict Time', marker="", linestyle=":", color='blue')
plt.plot(range(len(dqn_train_time_history)), dqn_train_time_history, label='DQN Train Time', marker="", linestyle="-", color='gray')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig('ql_learning_time.pdf', bbox_inches='tight')
plt.show()