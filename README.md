# LunarLander_DQN

## Deep Q-Network:
The code describes my implementation of a reinforcement learning Agent (here, a deep Q-learning agent) to solve the open ai environment “LunarLander-v2” with discrete values.
The goal of this Agent in the environment is to successfully land the Lunar-Lander with the correct speed and inclination from the observations it gets and his 4 possible actions. 

## Description of the environment:
The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Reward for moving from the top of the screen to the landing pad and zero speed is about 100.. 140 points. If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points. (https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py) <br>
The Agent receives 8 observations from its environments: horizontal and vertical coordinates, horizontal and vertical speed, angle, and angular speed if the legs have contact with the ground. 
From this observation space, it has an action space of 4 discrete actions it can take: do nothing, fire left, right, or the main engine.

## AgentDQN:
The Agent was implemented using the Keras TF backend. The structure of the model is 2 fully connected layers with 512 and 256 nodes that take the observation space as input and output the probability of the 4 possible actions. I used the Adam optimizer to compile the model and the mean squared error to calculate the loss function. <br>
The Agent selects his action based on an epsilon greedy policy with initial epsilon=1.0. After each episode, epsilon decays by * 0.995. The state, reward and done info are collected for this new action and are stored in the Agent memory as (s, a, r, s’). <br>
The Agent has a replay function allowing it to take a random sample from its memory with size (batch_size) to update his Q-learning.
I also implemented a maximum step size of 500 as it is sufficient for the model to land. <br>
An early stop is implemented if the mean rewards of the last episodes reached 200 and that epsilon is low enough (For this model it stopped after 616 iterations). A GIF of the model is taken each 50 episodes. <br>

The Agent created takes the following parameters:
- Initial epsilon: 1.0
- Epsilon decay: 0.995
- Epsilon minimum = 0.01
- Learning rate: 0.01
- Gamma: 0.99
- Memory = deque(maxlen=500000)
- Batch_size = 64
- Maximum steps = 500

### Model in the first episode
![movie0](https://user-images.githubusercontent.com/63811972/150113355-f6da2812-9e3c-4d65-af5b-34c8168d9ba1.gif)

### Model after 200 episodes
![movie200](https://user-images.githubusercontent.com/63811972/150113680-980c287e-3ccd-4ed8-a674-f0ba4eb68fb2.gif)

### Model after 400 episodes 
![movie400](https://user-images.githubusercontent.com/63811972/150114574-2f61eb9c-7c27-4dd8-898c-e26328865eb6.gif)

### Trained model after 616 episodes
![movie_trained_model_3](https://user-images.githubusercontent.com/63811972/150115760-8a1a76a8-90ca-4b88-9400-3f87fae225be.gif)

## Learning curve:

### Plot 1: Rewards
![Plot_616](https://user-images.githubusercontent.com/63811972/150115867-9dc40be4-a7c0-44c2-a45e-ae5266d57db0.png)

### Plot 2: Epsilon
![fig2](https://user-images.githubusercontent.com/63811972/150116462-6d1cc7fb-a2b7-4b8a-a575-3cbf645fd029.png)

Plot 1 represent the rewards earned at each episode played by the Agent. We can see that it starts to stabilize after 450 episodes when the epsilon is under 0.2 (plot 2) because the Agent takes fewer random actions, leading to better control and rewards in the environment.

We can conclude that the Agent has successfully learned the task to land with the correct position and speed.



