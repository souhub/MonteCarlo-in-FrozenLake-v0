import numpy as np


class MonteCarloAgent:
    def __init__(self, n_states, n_actions, epsilon, alpha, gamma) -> None:
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def reset(self):
        self.experiences = []

    # epsilon-greedy
    def take_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    # log rewards to learn
    def get_experience(self, state, action, reward):
        self.experiences.append(
            {'state': state, 'action': action, 'reward': reward})

    def learn(self):
        for i, exp in enumerate(self.experiences):
            G, t = 0, 0
            for j in range(i, len(self.experiences)):

                G += np.power(self.gamma, t)*self.experiences[j]['reward']
                t += 1
            state, action = exp['state'], exp['action']
            self.Q[state][action] = (1-self.alpha) * \
                self.Q[state][action]+self.alpha*G
