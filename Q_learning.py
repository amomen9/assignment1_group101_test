#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import time

import numpy as np
# from Helper import argmax, softmax
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

# Begin Class QLearningAgent ##########################################################################
class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # Modified by me:
        g_t = r + self.gamma * np.max(self.Q_sa[s_next]) * (0 if done else 1) # Compute the target return using the Q-learning update method
        self.Q_sa[s,a] += self.learning_rate * (g_t - self.Q_sa[s,a]) # Place the target return in this equation to acquire the new
        # value that Q-Value needs to be updated with. The new Q-value estimate for state s and action a
        # is obtained using the learning rate and the target return
        
# End Class QLearningAgent ##########################################################################


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    log_lines = []
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    relative_timesteps = 0
    step_pause = 0.3    # Plot update speed, in seconds
    
    
    # Modified by me:
    
    s = env.reset() # Initialize the environment
    
    for t in range(n_timesteps):
        a = agent.select_action(s, policy, epsilon, temp) # Select a specific action based on the defined policy
        s_next,r,done = env.step(a) # Take one step, and get the resulting next state, reward, and done signal from the step function of the environment
        agent.update(s,a,r,s_next,done) # Update agent's Q-values 
        s = s_next # Move on to the next state for the next iteration of the for loop
        if plot:
            env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=step_pause) # Plot the Q-value estimates during Q-learning execution
            #env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution
        
        if (t+1) % eval_interval == 0: # We need to see how many evaluation intervals fit
            # inside the number of steps. When interval is complete, evaluate the agent's performance on a separate evaluation environment and store the results for later observation
            eval_return = agent.evaluate(eval_env) # Evaluate the agent's performance on a separate evaluation environment
            eval_returns.append(eval_return) # Append the evaluation return to the list of the previous ones for later observation
            eval_timesteps.append(t+1) # Append the corresponding timestep for later observation
        
        if done: # If the episode is done (terminal state), reset the environment to start a new episode
            s = env.reset() # Reset the environment to start a new episode               
            exploration_info = f"Temp:, {temp}" if policy == "softmax" else (f"Epsilon:, {epsilon}" if policy == "egreedy" else "")
            log_lines.append(f"Goal after {t-relative_timesteps} timesteps, #Iteraion:, {t+1}, Policy:, {policy}" + (f", {exploration_info}" if exploration_info else "") + "\n")
            #print("Goal reached (terminal state) after {} timesteps. Absolute timesteps: {}. Policy: {}".format(t-relative_timesteps, t+1, policy))
            relative_timesteps = t

        
    with open("output_Q_learning.py.log", "a", encoding="utf-8") as f:
        f.writelines(log_lines)
        f.write("\n")
            
    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1
    
    
    # Exploration
    policy = 'egreedy' # 'greedy', 'egreedy', or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps=n_timesteps, learning_rate=learning_rate, gamma=gamma, policy=policy, epsilon=epsilon, temp=temp, plot=plot, eval_interval=eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
