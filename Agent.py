#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax
# from Helper import egreedy
import Environment


### One suggested simplest policy {eps, 1-eps} is below, however, I implement the one that was mentioned in the assignment instead.
#def egreedy(Qa_s, eps):
#    ''' Qa_s: vector of action values for state s
#        epsilon: exploration parameter '''
#    if np.random.rand() < eps:
#        return np.random.randint(0,len(Qa_s)) # Explore action space
#    else:
#        return argmax(Qa_s) # Exploit learned values

def egreedy(Qa_s, eps):
    """
    Sample one action using epsilon-greedy policy
    Qa_s: 1D array of Q-values for current state's actions
    eps: epsilon in the closed boundary [0,1]
    """
    n_A = len(Qa_s)     # number of actions
    greedy_a = argmax(Qa_s)  # tie breaking argmax()
    # Base probability for all actions, fill probs matrix with the same values (will not sum up to 1 yet)
    probs = np.full(n_A, eps / n_A, dtype=float)
    # Greedy action gets the remaining probability mass (1 - eps) plus its share of the exploration probability (eps/n_A)
    probs[greedy_a] = 1.0 - eps * (n_A - 1) / n_A
    selected_action = np.random.choice(n_A, p=probs)
    # Sample action from this distribution
    return int(selected_action)

def softmax(x, temp):   # aka Boltzmann policy (Mentioned as Boltzmann in the assignment)
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax
    probs = np.exp(z)/np.sum(np.exp(z)) # compute softmax
    selected_action = np.random.choice(len(x), p=probs) # Sample action from
    return int(selected_action)

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)



# Begin Class BaseAgent ##########################################################################
class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states       = n_states
        self.n_actions      = n_actions
        self.learning_rate  = learning_rate
        self.gamma          = gamma
        self.Q_sa           = np.zeros((n_states,n_actions))    # per assignment, initialize all Q-values to zero
        
    def select_action(self, s, policy='egreedy' , epsilon=None, temp=None):    # default policy is epsilon-greedy, but one can also select greedy or softmax (Boltzmann) policy
        
        if policy == 'greedy':
            # Modified by me:
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            a = int(argmax(self.Q_sa[s])) # Select the best known action to the agent (tie breaking argmax)
        
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # Modified by me:
            a = egreedy(self.Q_sa[s], epsilon) # Use the epsilon-greedy policy I implemented in Helper.py
                 
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # Modified by me:
            a = softmax(self.Q_sa[s], temp) # Selecting softmax        
        else:
            raise KeyError("Unknown policy type")
              
        return a
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns  = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s    = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
# End Class BaseAgent ##########################################################################
