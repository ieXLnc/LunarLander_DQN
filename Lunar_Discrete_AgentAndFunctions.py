import gym
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

from collections import deque

print('tf Version:', tf.__version__)
print('gym version:', gym.__version__)


# create class Agent
class AgentDQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay, name):

        self.env = env
        self.action_space = env.action_space
        self.num_action = self.action_space.n
        self.observation = env.observation_space
        self.observation_shape = self.observation.shape[0]
        self.count = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=500000)
        self.batch_size = 64
        self.name = name
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.observation_shape, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_action, activation='linear'))
        model.compile(optimizer=Adam(lr=self.lr), loss=mean_squared_error)
        model.summary()
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
             return random.randrange(self.num_action)

        prediction = self.model.predict(state)
        return np.argmax(prediction[0])

    def memorize(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def get_random_sample_from_replay_mem(self):
        random_sample = random.sample(self.memory, self.batch_size)
        return random_sample

    def update_count(self):
        # to replay a bit less because it breaks my computer
        self.count += 1
        update_count = 5
        self.count = self.count % update_count

    def replay(self):

        #  replay memory buffer size check
        if len(self.memory) < self.batch_size or self.count != 0:  # will replay once every 5 times after the inital 64
            return

        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, new_states, dones = self.unpack_random_sample(random_sample)
        expected_val = rewards + self.gamma * (np.amax(self.model.predict_on_batch(new_states), axis=1)) * (1 - dones)
        targets = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        targets[[indexes], [actions]] = expected_val

        self.model.fit(states, targets, epochs=1, verbose=0)

    def unpack_random_sample(self, random_sample):
        #  current_state, action, reward, new_state, done
        current_states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        new_states = np.array([i[3] for i in random_sample])
        dones = np.array([i[4] for i in random_sample])
        new_states = np.squeeze(new_states)
        current_states = np.squeeze(current_states)
        return current_states, actions, rewards, new_states, dones

    def save_model(self):
        save_dir = 'C:/Users/xavier/PycharmProjects/Q_Learning/models/'
        self.model.save(save_dir + self.name)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def gif(images, name, address="./recording/"):
        images[0].save(address + name, save_all=True, append_images=images[1:], optimize=True, duration=50, loop=0)


def plot_df(df, chart_name, title, x_label, y_label, color):
    plt.plot(df, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(chart_name)
    plt.close()


