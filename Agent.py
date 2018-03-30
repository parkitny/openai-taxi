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

    def to_string(self, var):
        #print("test", var)
        return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

    # Default vals
    def getAlpha(self):
        return 0.01

    def getGamma(self):
        return 1.0

    #Update policy Q
    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, state, episode, eps=None):
        epsilon = 1.0 / episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(env.nA) * epsilon / env.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
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
        
        #Some local vars for calc
        m_Q = self.Q#self.updatedPolicy()

        m_nA = self.nA
        m_State = state
        tmp = self.to_string(m_Q[state])
        print("POLICY ", tmp, "\n")
        print("ACTIONS ", self.nA)
        index = -1

        #return np.random.choice(self.nA)
        return np.argmax(m_Q[state])

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
        print("TAKING STEP")
        self.Q[state][action] += 1
