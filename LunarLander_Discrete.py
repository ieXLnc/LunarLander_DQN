import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

from Lunar_Discrete_AgentAndFunctions import AgentDQN
from Lunar_Discrete_AgentAndFunctions import gif, plot_df

print('Tf Version:', tf.__version__)
print('Gym version:', gym.__version__)


if __name__ == '__main__':
    # env
    env = gym.make('LunarLander-v2')
    TRAINED_MODEL = True
    NAME = 'LunarLander-v2_DISCRETE.h5'
    total_rewards = []

    if not TRAINED_MODEL:

        # initialize params
        EPSILON = 1.0
        LR = 0.001
        GAMMA = 0.99
        EPSILON_DECAY = 0.995
        NUM_EPISODES = 2000
        NUM_STEPS = 500
        env.seed(14)
        np.random.seed(14)
        epsilon_list = []

        Agent = AgentDQN(env, LR, GAMMA, EPSILON, EPSILON_DECAY, NAME)

        for episode in range(NUM_EPISODES):

            current_state = env.reset()
            current_state = np.reshape(current_state, [1, 8])
            rewards = 0
            images = []
            steps = 0

            for step in range(NUM_STEPS):
                env.render()

                if episode % 50 == 0:
                    # Render to frames buffer
                    image = (env.render(mode="rgb_array"))
                    image = Image.fromarray(image)
                    images.append(image)

                action = Agent.get_action(current_state)
                new_state, reward, done, _ = env.step(action)
                new_state = np.reshape(new_state, [1, 8])  # reshape to predict
                Agent.memorize(current_state, action, reward, new_state, done)
                rewards += reward

                current_state = new_state
                steps += 1

                Agent.update_count()
                Agent.replay()

                if done:
                    break

            epsilon_list.append(Agent.epsilon)

            Agent.update_epsilon()

            total_rewards.append(rewards)
            mean_reward = np.mean(total_rewards[-100:])
            print('Episode: {} | Steps taken: {} | Rewards: {} | Epsilon: {} | Mean_rewards: {}'.format(episode, steps, rewards, Agent.epsilon, mean_reward))

            # save model
            if episode % 25 == 0 and episode != 0:
                Agent.save_model()

            # early stop
            if mean_reward >= 200 and Agent.epsilon <= 0.1:
                Agent.save_model()
                print('Agent Trained... saved model {}, early stopped at {} episodes'.format(NAME, episode))
                break

            if episode % 50 == 0:
                name = 'movie' + str(episode) + '.gif'
                gif(images, name)

        epsilon_df = pd.DataFrame(epsilon_list)
        plot_df(epsilon_df, "./Plots/fig2.png", 'DQN Agent: epsilon decay', "Episode", "epsilon", 'green')

    elif TRAINED_MODEL:

        save_dir = "C:/Users/xavier/PycharmProjects/Q_Learning/models/"
        # load model
        trained_model = load_model(save_dir + NAME)
        print('Trained model loaded')
        NUM_EPISODES = 5

        print('Lauching...')

        for i in range(NUM_EPISODES):
            current_state = env.reset()
            num_observation_space = env.observation_space.shape[0]
            current_state = np.reshape(current_state, [1, num_observation_space])
            rewards = 0
            step_count = 600
            steps = 0
            images = []

            for step in range(step_count):
                env.render()

                if i == 1:
                    # Render to frames buffer
                    image = (env.render(mode="rgb_array"))
                    image = Image.fromarray(image)
                    images.append(image)

                selected_action = np.argmax(trained_model.predict(current_state)[0])
                new_state, reward, done, _ = env.step(selected_action)
                new_state = np.reshape(new_state, [1, num_observation_space])
                current_state = new_state
                rewards += reward

                if done:
                    break

                steps += 1

            total_rewards.append(rewards)

            if i == 1:
                name = 'movie' + '_trained_model' + '.gif'
                gif(images, name)

            print('Episode: {} | Steps taken: {} | Rewards: {}'.format(i+1, steps, rewards))

    env.close()

    # plot the rewards and epsilon
    reward_df = pd.DataFrame(total_rewards)
    plot_df(reward_df, "./Plots/fig1.png", 'DQN Agent: rewards in function of episodes', "Episode", "Reward", 'red')
