from deepQAgent import Agent
from unityagents import UnityEnvironment
from collections import deque
import matplotlib.pyplot as plt
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
#agent.load_checkpoint("agent_score_22.h5")
epochs = 1000
max_score = -100000
movingAvg = deque(maxlen = 100)
steps = 0
rewards_plot = []
for epoch in range(epochs):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
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
        if steps%4 == 0:
            agent.train() #train the network
        if done:                                       # exit loop if episode finished
            break
    movingAvg.append(score)
    scoreavg = sum(movingAvg)/len(movingAvg)
    rewards_plot.append(scoreavg)
    if scoreavg > max_score:
        max_score = scoreavg
        print("**************************************************************")
        print(epoch, "New max score:", scoreavg)
        agent.save("checkpoint.pth")
        print("**************************************************************")
    print("Epoch:", epoch, "Score: ", score, scoreavg, end = "\r")

plt.xlabel("Epochs")
plt.ylabel("Avg reward of last 100 episodes")
plt.plot(rewards_plot)
plt.show()