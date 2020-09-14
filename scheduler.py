# Created by Chenye Yang on 2020/8/12.
# Copyright © 2020 Chenye Yang. All rights reserved.

import random
import numpy as np
import matplotlib.pyplot as plt

class Scheduler():
    def epsilon_greedy(self, value, e, seed=None):
        '''
        Implement Epsilon-Greedy policy.

        Inputs:
        value: numpy ndarray
        A vector of values of actions to choose from
        e: float
        Epsilon
        seed: None or int
        Assign an integer value to remove the randomness

        Outputs:
        action: int
        Index of the chosen action
        '''
        assert len(value.shape) == 1
        assert 0 <= e <= 1

        if seed != None:
            np.random.seed(seed)

        ############################
        # YOUR CODE STARTS HERE

        if random.random() > e:  # prob = 1-e
            action = np.argmax(value)
        else:  # prob = e
            action = random.randrange(value.size)

        # YOUR CODE ENDS HERE
        ############################
        return action


    def QLearning(self, env, num_episodes, gamma, lr, e):
        """
        Implement the Q-learning algorithm following the epsilon-greedy exploration.
        Inputs:
        env: OpenAI Gym environment
                env.P: dictionary
                        P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                        probability: float
                        nextstate: int
                        reward: float
                        terminal: boolean
                env.nS: int
                        number of states
                env.nA: int
                        number of actions
        num_episodes: int
                Number of episodes of training
        gamma: float
                Discount factor.
        lr: float
                Learning rate.
        e: float
                Epsilon value used in the epsilon-greedy method.
        Outputs:
        Q: numpy.ndarray
        """
        Q = np.zeros((env.nS, env.nA))

        ############################
        # YOUR CODE STARTS HERE

        max_iteration = 600
        average_reward_Q = [0 for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = random.randint(0, (env.nS - 1))  # randomly initialize start state
            step = 0

            while (not env.P[state][0][0][3] and step < max_iteration):
                step += 1
                action = self.epsilon_greedy(Q[state], e)  # choose action from State using e-greedy policy derived from Q
                reward = env.P[state][action][0][2]  # take action, get reward
                next_state = env.P[state][action][0][1]  # take action, get next state
                diff = Q[next_state]
                Q[state][action] = Q[state][action] + lr * (
                            reward + gamma * Q[next_state][np.argmax(Q[next_state])] - Q[state][action])
                state = next_state
                average_reward_Q[episode] += reward

            if step != 0:
                average_reward_Q[episode] = average_reward_Q[episode] / step

        # Plot the learning process of both algorithms for training 1000 episodes.
        plt.plot([i for i in range(num_episodes)], average_reward_Q, label="Q-Learning")
        plt.legend()
        plt.xlabel('episodes numbers')
        plt.ylabel('average rewards')
        plt.title('Q-Learning learning process')
        plt.show()

        # YOUR CODE ENDS HERE
        ############################

        return Q

    def SARSA(self, env, num_episodes, gamma, lr, e):
        """
        Implement the SARSA algorithm following epsilon-greedy exploration.
        Inputs:
        env: OpenAI Gym environment
                env.P: dictionary
                        P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                        probability: float
                        nextstate: int
                        reward: float
                        terminal: boolean
                env.nS: int
                        number of states
                env.nA: int
                        number of actions
        num_episodes: int
                Number of episodes of training
        gamma: float
                Discount factor.
        lr: float
                Learning rate.
        e: float
                Epsilon value used in the epsilon-greedy method.
        Outputs:
        Q: numpy.ndarray
                State-action values
        """
        Q = np.zeros((env.nS, env.nA))
        ############################
        # YOUR CODE STARTS HERE

        max_iteration = 600
        average_reward_SARSA = [0 for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = random.randint(0, (env.nS - 1))  # randomly initialize start state
            action = self.epsilon_greedy(Q[state], e)  # choose action from State using e-greedy policy derived from Q
            step = 0

            while (not env.P[state][0][0][3] and step < max_iteration):
                step += 1
                reward = env.P[state][action][0][2]  # take action, get reward
                next_state = env.P[state][action][0][1]  # take action, get next state
                next_action = self.epsilon_greedy(Q[next_state], e)
                diff = [Q[next_state][a] - Q[state][action] for a in range(env.nA)]
                Q[state][action] = Q[state][action] + lr * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                state = next_state
                action = next_action
                average_reward_SARSA[episode] += reward

            if step != 0:
                average_reward_SARSA[episode] = average_reward_SARSA[episode] / step

        # Plot the learning process of both algorithms for training 1000 episodes.
        plt.plot([i for i in range(num_episodes)], average_reward_SARSA, label="SARSA")
        plt.legend()
        plt.xlabel('episodes numbers')
        plt.ylabel('average rewards')
        plt.title('SARSA learning process')
        plt.show()

        # YOUR CODE ENDS HERE
        ############################

        return Q

class Scheduler_Dynamic():
    def epsilon_greedy(self, value, e, seed=None):
        '''
        Implement Epsilon-Greedy policy.

        Inputs:
        value: numpy ndarray
        A vector of values of actions to choose from
        e: float
        Epsilon
        seed: None or int
        Assign an integer value to remove the randomness

        Outputs:
        action: int
        Index of the chosen action
        '''
        assert len(value.shape) == 1
        assert 0 <= e <= 1

        if seed != None:
            np.random.seed(seed)

        ############################
        # YOUR CODE STARTS HERE

        if random.random() > e:  # prob = 1-e
            action = np.argmax(value)
        else:  # prob = e
            action = random.randrange(value.size)

        # YOUR CODE ENDS HERE
        ############################
        return action


    def QLearning(self, env, num_episodes, gamma, lr, e):
        """
        Implement the Q-learning algorithm following the epsilon-greedy exploration.
        Inputs:
        env: OpenAI Gym environment
                env.P: dictionary
                        P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                        probability: float
                        nextstate: int
                        reward: float
                        terminal: boolean
                env.nS: int
                        number of states
                env.nA: int
                        number of actions
        num_episodes: int
                Number of episodes of training
        gamma: float
                Discount factor.
        lr: float
                Learning rate.
        e: float
                Epsilon value used in the epsilon-greedy method.
        Outputs:
        Q: numpy.ndarray
        """
        Q = np.zeros((env.nS, env.nA))

        ############################
        # YOUR CODE STARTS HERE

        max_iteration = 600
        average_reward_Q = [0 for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = random.randint(0, (env.nS - 1))  # randomly initialize start state
            step = 0

            while (step < max_iteration):
                step += 1
                action = self.epsilon_greedy(Q[state], e)  # choose action from State using e-greedy policy derived from Q
                next_state, reward = env.trans(state)[action][0][1:3]  # take action, get next state and reward
                diff = Q[next_state]
                Q[state][action] = Q[state][action] + lr * (
                            reward + gamma * Q[next_state][np.argmax(Q[next_state])] - Q[state][action])
                state = next_state
                average_reward_Q[episode] += reward

            if step != 0:
                average_reward_Q[episode] = average_reward_Q[episode] / step

        # Plot the learning process of both algorithms for training 1000 episodes.
        plt.plot([i for i in range(num_episodes)], average_reward_Q, label="Q-Learning")
        plt.legend()
        plt.xlabel('episodes numbers')
        plt.ylabel('average rewards')
        plt.title('Q-Learning learning process')
        plt.show()

        # YOUR CODE ENDS HERE
        ############################

        return Q

    def SARSA(self, env, num_episodes, gamma, lr, e):
        """
        Implement the SARSA algorithm following epsilon-greedy exploration.
        Inputs:
        env: OpenAI Gym environment
                env.trans(): dictionary
                        trans(state)[action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                        probability: float
                        nextstate: int
                        reward: float
                        terminal: boolean
                env.nS: int
                        number of states
                env.nA: int
                        number of actions
        num_episodes: int
                Number of episodes of training
        gamma: float
                Discount factor.
        lr: float
                Learning rate.
        e: float
                Epsilon value used in the epsilon-greedy method.
        Outputs:
        Q: numpy.ndarray
                State-action values
        """
        Q = np.zeros((env.nS, env.nA))
        ############################
        # YOUR CODE STARTS HERE

        max_iteration = 600
        average_reward_SARSA = [0 for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = random.randint(0, (env.nS - 1))  # randomly initialize start state
            action = self.epsilon_greedy(Q[state], e)  # choose action from State using e-greedy policy derived from Q
            step = 0

            while (step < max_iteration):
                step += 1
                next_state, reward = env.trans(state)[action][0][1:3]  # take action, get next state and reward
                next_action = self.epsilon_greedy(Q[next_state], e)
                diff = [Q[next_state][a] - Q[state][action] for a in range(env.nA)]
                Q[state][action] = Q[state][action] + lr * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                state = next_state
                action = next_action
                average_reward_SARSA[episode] += reward

            if step != 0:
                average_reward_SARSA[episode] = average_reward_SARSA[episode] / step

        # Plot the learning process of both algorithms for training 1000 episodes.
        plt.plot([i for i in range(num_episodes)], average_reward_SARSA, label="SARSA")
        plt.legend()
        plt.xlabel('episodes numbers')
        plt.ylabel('average rewards')
        plt.title('SARSA learning process')
        plt.show()

        # YOUR CODE ENDS HERE
        ############################

        return Q

if __name__ == '__main__':
    # test
    import gym
    from helpers import utils
    from helpers import evaluation_utils

    env = gym.make('Taxi-v3')
    scheduler = Scheduler()

    Q = scheduler.QLearning(env=env.env, num_episodes=1000, gamma=1, lr=0.1, e=0.1)
    print('Action values:')
    print(Q)
    policy_estimate = utils.action_selection(Q)
    evaluation_utils.render_episode(env, policy_estimate)

    Q = scheduler.SARSA(env=env.env, num_episodes=1000, gamma=1, lr=0.1, e=0.1)
    print('Action values:')
    print(Q)
    policy_estimate = utils.action_selection(Q)
    evaluation_utils.render_episode(env, policy_estimate)