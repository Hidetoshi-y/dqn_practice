import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'SpaceInvaders-v0'

# get environment
env = gym.make(ENV_NAME)
print(env.action_space.n)
print(env.action_space)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# define network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions)) 
model.add(Activation('linear'))
print(model.summary())

# parameter configure
memory = SequentialMemory(limit=60000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# fit
#dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# save weught
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# test 5 episodes
dqn.test(env, nb_episodes=5, visualize=True)
#dqn.test(env, nb_episodes=5, visualize=False)