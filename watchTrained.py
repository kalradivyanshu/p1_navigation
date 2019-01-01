from deepQAgent import Agent
from unityagents import UnityEnvironment
from collections import deque

env = UnityEnvironment(file_name="Banana.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', list(state))
state_size = len(state)
print('States have length:', state_size)

agent = Agent()
agent.load_checkpoint("checkpoint.pth") #load checkpoint
agent.randomness = 0
epochs = 10000
max_score = -100000
movingAvg = deque(maxlen = 100)
steps = 0
for epoch in range(10):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.action(state)       # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.remember(state, next_state, action, reward, done)
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        steps += 1
        if done:                                       # exit loop if episode finished
            break