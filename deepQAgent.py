import torch.nn as nn
from collections import deque
import random
import numpy as np
from network import Net
import torch
import torch.optim as optim


class Agent:
    """Interacts with and learns from the environment."""
    def __init__(self):
        self.actions = 4
        self.state_space = 37
        self.memory = deque(maxlen = 100000)
        self.randomness = 1
        self.least_randomness = 0.01
        self.randomness_decay = 0.995
        self.gamma = 0.95
        self.network = Net()
        self.qtarget_net = Net()
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        self.loss_func = nn.MSELoss()
        self.tau = 1e-3


    def remember(self, state, next_state, action, reward, done):
        """stores state, next_state, action, reward and done"""
        self.memory.append([state, next_state, action, reward, done])
    
    def action(self, state):
        """returns an action that has to be taken given the state"""
        if random.random() < self.randomness:
            return random.randint(0, self.actions - 1)
        state = torch.from_numpy(np.array(state).reshape(1, self.state_space)).float()
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()
        #
        return np.argmax(action_values.detach().numpy())

    def save(self, name):
        """save the model"""
        torch.save(self.network.state_dict(), name)
    
    def load_checkpoint(self, name):
        """load the model from memory"""
        self.network.load_state_dict(torch.load(name))

    def train(self, batch_size = 128):
        """trains the network based on previous experiences"""
        try:
            memory_buffer = random.choices(self.memory, k = batch_size)
        except:
            memory_buffer = self.memory
        predictions = []
        actuals = []
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for state, next_state, action, reward, done in memory_buffer:
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        states, next_states, actions, rewards, dones = [np.array(x).astype(np.float32) for x in [states, next_states, actions, rewards, dones]]
        #print(actions.shape)
        dones = dones.astype(np.uint8)
        states, next_states, actions, rewards, dones = [torch.from_numpy(x).float() for x in [states, next_states, actions, rewards, dones]]
        actions = actions.view(-1, 1)
        rewards = rewards.view(-1, 1)
        #print(actions.shape, states.shape)
        dones = dones.view(-1, 1)
        #print([x.shape for x in [states, actions, rewards, next_states, dones]])
        Q_targets_next = self.qtarget_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.network(states).gather(1, actions.long())
        #print(Q_expected.shape, Q_targets.shape)
        loss = self.loss_func(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.randomness = max(self.least_randomness, self.randomness*self.randomness_decay)
        for target_param, local_param in zip(self.qtarget_net.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

if __name__ == "__main__":
    state = [1.0, 0.0, 0.0, 0.0, 0.8440813422203064, 0.0, 0.0, 1.0, 0.0, 0.07484719902276993, 0.0, 1.0, 0.0, 0.0, 0.2575500011444092, 1.0, 0.0, 0.0, 0.0, 0.7417734265327454, 0.0, 1.0, 0.0, 0.0, 0.25854846835136414, 0.0, 0.0, 1.0, 0.0, 0.0935567170381546, 0.0, 1.0, 0.0, 0.0, 0.3196934461593628, 0.0, 0.0]
    agent = Agent()
    agent.action(state)