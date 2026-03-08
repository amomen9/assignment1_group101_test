#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import pandas as pa
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # begin own code
        G = 0
        for t in range(len(actions)-1, -1, -1):
            G = rewards[t] + self.gamma * G
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]]) # compute the state-action value function
        # end own code


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # begin own code
    timestep = 0
    last_eval = -eval_interval # initiate when to run evaluations

    while timestep < n_timesteps:
        s = env.reset() # resets the environment and obtains starting state s
        states = [s] # create the lists of observed states, action and rewards observed in the episode
        actions = []
        rewards = []

        for t in range(max_episode_length): # collect episode
            timestep += 1

            a = pi.select_action(s, policy, epsilon, temp) # sample action
            s_next, r, done = env.step(a) # simulate environment

            states.append(s_next) # update lists
            actions.append(a)
            rewards.append(r)

            if done:
                break # terminal
            else:
                s = s_next # next iteration

            if timestep - last_eval >= eval_interval:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(timestep)
                last_eval = timestep # update when the last evaluation was

        pi.update(states, actions, rewards) # execute the update with discounted rewards
     # end own code
    
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution
                 
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
