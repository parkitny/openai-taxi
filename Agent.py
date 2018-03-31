import numpy as np
from collections import defaultdict


#
# Agent implementation with SARSA control
#
class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.i_episode = 1
        self.currentPolicy_s = self.epsilon_greedy_probs(self.Q[ 0], self.i_episode, 0.005)

    def to_string(self, var):
        #print("test", var)
        return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

    # Default vals
    def getAlpha(self):
        return 0.01

    def getGamma(self):
        return 0.85

    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        return Qsa + np.float64(alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, Q_s, episode, eps=None):
        epsilon = 1.0 / episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.i_episode += 1
        Q = self.Q
        
        self.currentPolicy_s = self.epsilon_greedy_probs(self.Q[next_state], self.i_episode, 0.005)
        newQ = self.update_Q(Q[state][action], \
                          np.dot(Q[next_state], self.currentPolicy_s), \
                          reward, \
                          self.getAlpha(), \
                          self.getGamma())
        self.Q[state][action] = newQ
