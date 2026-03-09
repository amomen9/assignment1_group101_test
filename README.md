# Dynamic programming
To test how well the Dynamic Programming worked the following scenarios were tested:

Q value iteration at iterations 0, 8 (halfway) and 16 (at convergence) with
goal_location = [7,3]
gamma         = 0.99
threshold     = 0.001

Seeing what happens with a new goal location. The parameters used are:
goal_location = [6,2]
gamma         = 0.99
threshold     = 0.001

Seeing what happens with a lower discount factor. The parameters used are:
goal_location = [7,3]
gamma         = 0.5
threshold     = 0.001

Seeing what happens with a higher threshold. The parameters used are:
goal_location = [7,3]
gamma         = 0.99
threshold     = 78.

These experiments are all run consecutively using
python DynamicProgramming.py
The plots will also be added to the DynamicProgramming_plots directory that will be generated
