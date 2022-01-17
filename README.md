# Markov-Decision-Process-GridWorld
Implementing MDP in a customizable Grid World (Value and Policy Iteration). (Python 3)

Grid World is a scenario where the agent lives in a grid. The grid has m by n dimension which contains terminal states, walls, rewards and transition probabilities. Transition Probabilities exist in order to introduce stochasticity in the motion of the agent, and Rewards could be considered as punishments for the agent for every step that it takes. A reward which punishes the agent mildly for every step taken forces the agent to reach the terminal states with minimum total steps taken. 

The data is fed to the program from an Input file. The file contains details of the puzzle, and is customizable. The program performs both Value Iteration and Policy iteration to attain the Optimal Policies for a specified Reward.
