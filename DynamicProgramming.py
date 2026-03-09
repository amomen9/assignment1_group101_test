#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import os
import matplotlib.pyplot as plt

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.gamma     = gamma
        self.Q_sa      = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 

        "Use the provided argmax function to select the action corresponding to the maximum of Q(s,a)"
        return argmax(self.Q_sa[s])
        
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''

        """
        The Q table with elements Q[s,a] has states along the rows and actions along the columns.
        The term gamma * max_a' Q(s',a') represents the maximum value of Q by considering the different actions available
        In order to code this we check the maximum of self.Q_sa along axis 1 (checking along a row). For example
        if we have the following matrix
        [1  2   3]
        [9  1   2]
        [10 12  8]
        and use np.max() along axis 1 we get the following array: [3, 9, 12]. Along axis 0 we would get the following array
        [10, 12, 8] (checking along the columns)
        """
        self.Q_sa[s,a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa,axis=1))) #?

    
def Q_value_iteration(env, gamma=1.0, threshold=0.001, stop="end"):
    """
    Runs Q-value iteration. Returns a converged QValueIterationAgent object

    env         : environment
    gamma       : float. Discount factor
    threshold   : float. 
    stop        : str. When to stop the value iteration. Should be either ['begin','halfway','end']
    """

    "Initialise the QValueIterationAgent class"
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    counter = 1
    "As long as the maximum number of iterations is not reached, keep repeating"
    while counter < 100:
        "Reset the error to zero"
        error   = 0 
        
        "If we want to stop the Q-value iteration algorithm, this handles that"
        match stop:
            case "begin":
                if counter == 1:
                    break
            case "halfway":
                if counter == 6: # While testing it takes about 12 iterations to converge with a threshold of 70. and gamma of 0.99
                    break
            case "end":
                "Let the algorithm converge naturally"
                pass

        "For all states"
        for s in range(env.n_states):
            "For all actions"
            for a in range(env.n_actions):
                "Store current estimate"
                x = QIagent.Q_sa[s,a]

                "Get p_sas and r_sas"
                p_sas, r_sas = env.model(s,a)
    
                "Update the Q-table"
                QIagent.update(s,a,p_sas,r_sas)

                "Update the error"
                error = np.max([error,np.abs(x - QIagent.Q_sa[s,a])])

        print("Q-value iteration, iteration {}, max error {}".format(counter,error))

        "Stop the algorithm if the convergence criterion has been reached"
        if error < threshold:
            print(f"Algorithm finished after {counter} iterations by convergence")
            break

        "Increase the iteration counter"
        counter += 1


    return QIagent

def experiment(gamma, threshold, stop, path, goal = [[7,3]],title=""):
    """
    Run an experiment with the given parameters.

    gamma       : float. Discount factor
    threshold   : float. 
    stop        : str. Either begin, halfway or end. Decides when to store the Q-table.
                       Assumes it takes 16 iterations to finish. This is the case
                       for gamma=0.99 amd threshold=0.001
    path        : str. Place to store the figures
    """

    "Initialise the environment"
    env       = StochasticWindyGridworld(initialize_model=True)

    "Set the goal location"
    env.change_goal_location(goal)

    "Run the model construction again so the goal location is handled well and the r_sas and p_sas models are updated"
    env._construct_model()
    env.render() 

    "Set the title of the figure"
    env.ax.set_title(title)

    "Make sure the figure has a better layout"
    plt.tight_layout()

    "This parameter decides whether we want to create a plot of the environment at different stages of the VI algorithm"
    # stop      = None #can be set to either ['begin, 'halfway','end']. Any other input will result in no stopping
    QIagent   = Q_value_iteration(env,gamma,threshold,stop=stop)
    
    "view optimal policy"
    done      = False
    s         = env.reset()

    "Store the rewards and states in arrays"
    states    = np.array([])
    rewards   = np.array([])

    "Save the figure at the beginning"
    step = 1
    while not done:
        "Select action to take"
        a               = QIagent.select_action(s)

        "Take this action and recieve the next state, the reward and whether the terminal state is reached"
        s_next, r, done = env.step(a)

        "Store the state"
        states          = np.append(states,s)

        "Store the reward"
        rewards         = np.append(rewards,r)

        "Plot the process and include an arrow in the plot that indicates the greedy action in a state"
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)

        "Save a figure"
        if step == 1:
            match stop:
                case "begin":
                    "Store the figure"
                    env.fig.savefig(f"{path}")

                    "Set done to true, because it will never get there taking random actions"
                    done = True

                case "halfway":
                    env.fig.savefig(f"{path}")
                case "end":
                    env.fig.savefig(f"{path}")
        "Increase the step counter so we only store the figure once"
        step += 1           
        
        "Update the current state"
        s     = s_next
    
    "Close the figure once we're done"
    plt.close(env.fig)
    
    "We can calculate V(s) using the rewards array"
    def value_function(s,gamma=gamma):
        """
        Returns the cumulative reward from a given state s
        assuming that the optimal policy is being followed
        """

        "This function only works for the states that are visited and not for others"
        assert s in states , f"Select a state from one of these:\n{states}"

        "Find the index of the given state"
        index = np.where(states == s)[0][0]

        "Let the user know what this state corresponds to"
        print(f"State {s} corresponds to the location {env._state_to_location(s)}")

        """
        Suppose that after state s follows a number of states until the terminal
        state T is reached. In that case the value function can be determined using
        V(s) = sum_(t = 0)^(T-1) gamma^T r_T)
        """
        
        "Apply this equation"
        return np.sum(rewards[index:]* gamma**np.arange(len(rewards[index:])))
    
    "The number of steps is equal to the number of rewards received"
    number_of_steps          = len(rewards)

    """
    Calculate the mean reward received per timestep. Here gamma is set to one
    to take future rewards into account equally. If gamma is set to <1 then
    determining the mean_reward_per_timestep using the value function would
    not be accurate. In that case, this could instead be determined as follows:
    
    mean_reward_per_timestep = (100 + (number_of_steps - 1) * -1) / number_of_steps

    This is the  case because the agent will definitely find a reward of 100. It will
    get that reward in its final step. All other steps incur a reward of negative 1.
    Sum those two quantities and divide them by two to get the mean reward per timestep.
    This requires knowledge of the rewards gained in the environment and assumes all
    steps other than the one to the terminal state incur the same reward.
    """
    mean_reward_per_timestep = value_function(3,gamma=1) / number_of_steps
    
    print("Number of steps is",number_of_steps)
    print(f"V({env._location_to_state((0,3))}) =",value_function(env._location_to_state((0,3)),gamma))
    print("Mean reward per timestep under optimal policy: {:.5g}".format(mean_reward_per_timestep))

def final_experiment():
    """
    This function will be used to run the experiments which can be used in the report
    or for the grader
    """
    
    "Generate a folder to store plots in, if it doesn't exist already"
    plot_dir = "DynamicProgramming_plots"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    """
    Run the experiments with the different stages of 
    the value iteration algorithm
    """
    # print(25*"- ")
    # experiment(0.99,70.,"begin",f"{plot_dir}/step1_VI_at_begining.pdf")
    # print(25*"- ")
    # experiment(0.99,70.,"halfway",f"{plot_dir}/step1_VI_halfway.pdf")
    # print(25*"- ")
    # experiment(0.99,70.,"end",f"{plot_dir}/step1_VI_at_end.pdf")
    # print(25*"- ")

    "Run the experiment at a different goal location"
    experiment(0.99,1.,"end",f"{plot_dir}/new_goal.pdf",[[6,2]],"Goal location is now at (6,2)")

    "Run the experiment at the original goal location but with a gamma of 1"
    experiment(0.5,70.,"end",f"{plot_dir}/gamma_0_5.pdf",title=r"Discount factor is now $\gamma = 0.5$")

    "Run the experiment at the original goal location but with a higher threshold"
    experiment(0.99,78.,"end",f"{plot_dir}/high_threshold.pdf",title=r"Threshold is now $\eta = 78.$")

    
if __name__ == '__main__':
    # experiment(0.99,70.,None,"")
    # experiment(0.99,0.001,"begin","DynamicProgramming_plots")
    # experiment(0.99,0.001,None,"DynamicProgramming_plots",[[2,3]],"Test title")
    final_experiment()


