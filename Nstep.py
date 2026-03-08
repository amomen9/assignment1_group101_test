#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent
from Helper import LearningCurvePlot, smooth

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        # begin own code
        for t in range(len(actions)):
            disc_r_list = [] # collect discounted rewards

            for i in range(min(n, len(rewards)-t)):
                disc_r = self.gamma**i * rewards[t+i] # calculate discounted reward
                disc_r_list.append(disc_r)
        
            G = np.sum(disc_r_list) # terminal
            if t+n < len(states) and not done:  # prevent it from going past the end of the episode
                G += self.gamma**n * np.max(self.Q_sa[states[t+n]]) # target, taking n steps into account

            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]]) # compute the state-action value function
        # end own code


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
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

            if timestep - last_eval >= eval_interval:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(timestep)
                last_eval = timestep # update when the last evaluation was

            if done:
                break # terminal
            else:
                s = s_next # next iteration

        pi.update(states, actions, rewards, done, n) # execute the update with discounted rewards
     # end own code

        if plot:
           env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
  
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100    
    gamma = 1.0
    learning_rate = 0.1
    n = 5 # [1, 3, 10]
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, policy, epsilon, temp, plot, n=n)

    # begin own code
    # LCPlot = LearningCurvePlot(title="Learning Curve n-step Q-learning") # creating plot

    # for n in n_list:

    #     n_step_list = []
    #     for i in range(20): # 20 repetitions
    #         eval_returns, eval_timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, policy, epsilon, temp, plot, n=n)
    #         print(len(eval_returns))
    #         n_step_list.append(eval_returns)

    #     n_step_mean = np.mean(n_step_list, axis=0) # averaging over 20 repetitions

    #     # plotting with Helper
    #     LCPlot.add_curve(eval_timesteps, n_step_mean, label=f"n={n}") # smooth(sarsa_returns, window=11)

    # LCPlot.save(name="learning_curve_n_step_q_learning.pdf")
    # end own code
    
if __name__ == '__main__':
    test()
