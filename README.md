# JSSP

Job Shop Scheduling Problem is a widely studied combinatorial, NP hard optimization problem. The aim of the problem is to allocating shared resources over time to competing activities in order to reduce the overall time (makespan) needed to complete all activities. We have n number of Jobs and m number of Machines and these jobs need to be completed using these machines. Each job has its own sequence or specific order in which it needs to be completed. Aim of this project is to determine the least makespan using Reinforcement Learning. 

The Jssp.py file contains the Job Shop Scheduling environment. The action space used is Multi Discrete. The environment has been validated.

Stable baseline 3 Library is used in this project. Upon using different algorithms, Proximal Policy Optimization (PPO) has been deemed as the algorithm to be used in order to achieve the best results.
