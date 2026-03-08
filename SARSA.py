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
from Q_learning import q_learning
from Helper import LearningCurvePlot, smooth

class SarsaAgent(BaseAgent):
        
    def update(self, s, a, r, s_next, a_next, done):

        # begin own code
        if done:
            G = r # for the terminal state, there is no future reward to estimate
        else:
            G = r + self.gamma * self.Q_sa[s_next, a_next] # target value as the direct reward plus the taken (no max) discounted future rewards

        self.Q_sa[s, a] += self.learning_rate * (G - self.Q_sa[s, a]) # compute the state-action value function
        # end own code

   
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # begin own code
    s = env.reset() # resets the environment and obtains starting state s
    a = pi.select_action(s, policy, epsilon, temp) # sample first action a from the policy

    timestep = 0
    last_eval = -eval_interval # initiate when to run evaluations

    while timestep < n_timesteps:
        timestep += 1

        s_next, r, done = env.step(a) # samples the next state and reward from the environment
        a_next = pi.select_action(s_next, policy, epsilon, temp) # get the next action through the policy

        pi.update(s, a, r, s_next, a_next, done) # tabular learning update

        if done:
            s = env.reset() # resets environment, new s
            a = pi.select_action(s, policy, epsilon, temp) # new a
        else:
            s = s_next # s for next iteration
            a = a_next # a for next iteration

        if timestep - last_eval >= eval_interval:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(timestep)
            last_eval = timestep # update when the last evaluation was
    # end own code

        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1 # [0.03, 0.1, 0.3]

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)

    # # begin own code
    # LCPlot = LearningCurvePlot(title="Learning Curve SARSA vs Q-learning") # creating plot

    # for lr in learning_rates:

    #     sarsa_list = []
    #     for i in range(20): # 20 repetitions
    #         eval_returns, eval_timesteps = sarsa(n_timesteps, lr, gamma, policy, epsilon, temp, plot)
    #         print(len(eval_returns))
    #         sarsa_list.append(eval_returns)

    #     q_learning_list = [] # compare sarsa to q_learning
    #     for i in range(20): # 20 repetitions
    #         eval_returns, eval_timesteps = q_learning(n_timesteps, lr, gamma, policy, epsilon, temp, plot)
    #         q_learning_list.append(eval_returns)
    
    #     sarsa_mean = np.mean(sarsa_list, axis=0) # averaging over 20 repetitions
    #     q_learning_mean = np.mean(q_learning_list, axis=0)

    #     # plotting with Helper
    #     LCPlot.add_curve(eval_timesteps, sarsa_mean, label=f"SARSA, lr={lr}") # smooth(sarsa_returns, window=11)
    #     LCPlot.add_curve(eval_timesteps, q_learning_mean, label=f"Q-learning, lr={lr}")

    # LCPlot.save(name="learning_curve_sarsa_vs_q_learning.pdf")
    # end own code
            
if __name__ == '__main__':
    test()
