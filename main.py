import gym
import numpy as np

from agent import MonteCarloAgent
from utils import show_success_rate

EPSILON = 0.2
ALPHA = 0.2
GAMMA = 0.99
EPISODES = 20000
ENV = 'FrozenLake-v0'


def main():
    env = gym.make(ENV)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = MonteCarloAgent(n_states, n_actions, EPSILON, ALPHA, GAMMA)
    episode_rewards = np.zeros((EPISODES))

    for e in range(EPISODES):
        state = env.reset()
        agent.reset()
        done = False

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            agent.get_experience(state, action, reward)
            state = next_state
            episode_rewards[e] += reward

        agent.learn()

        if(e % 1000 == 0 and e != 0):
            print(f'{e}/{EPISODES} episode completed')

    return episode_rewards


if __name__ == '__main__':
    episode_rewards = main()
    show_success_rate(episode_rewards)
