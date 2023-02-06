import gym
import numpy as np
from JsspEnvironment.envs.Jssp import Jssp

class Validate:
    def __init__(self, instance_id="ft06", makespan=55, roll_out=110):

        self.env = gym.make('Jssp-v0', instance_id=instance_id, hyper_parameters=None,
                       roll_out_timestep=roll_out)
        self.makespan = makespan
        self.solution = self._solution_file()
        self.instance_id = instance_id
        self.roll_out = roll_out
        self._start_validation()


    def _solution_file(self):

        '''solution_sequence = np.array([[8, 7, 2, 5, 10, 1, 6, 4, 3, 9],
                                      [1, 8, 4, 3, 7, 5, 2, 9, 10, 6],
          [6, 3, 2, 7, 10, 5, 4, 1, 9, 8],
          [3, 6, 7, 8, 4, 10, 1, 9, 5, 2],
          [2, 9, 3, 8, 10, 6, 1, 5, 4, 7],
          [2, 6, 5, 7, 8, 9, 1, 4, 10, 3],
          [1, 8, 7, 5, 6, 3, 4, 10, 9, 2],
          [2, 1, 4, 3, 5, 10, 9, 8, 6, 7],
          [10, 3, 1, 7, 8, 4, 9, 2, 6, 5],
          [10, 1, 2, 6, 7, 3, 8, 4, 5, 9]], dtype=int)'''

        solution_sequence = np.array([[4, 3, 1,	6, 2, 5],
                                      [2, 4, 6,	5, 1, 3],
                                      [1, 3, 2,	5, 4, 6],
                                      [3, 6, 4,	1, 2, 5],
                                      [2, 5, 4,	3, 6, 1],
                                      [3, 6, 2,	5, 1, 4]], dtype=int)
        return solution_sequence

    def _start_validation(self):
        self.env.reset()
        done = False
        action = 0
        while not done:

            '''if self.env.operation_progression[self.env.machine_index] < self.env.jobs:
                            action = self.solution[self.env.machine_index][
                                    self.env.operation_progression[self.env.machine_index]]'''
            action = []
            for machine in range(self.env.machines):

                if self.env.operation_progression[machine] != self.env.machines:
                    action.append(self.solution[machine][self.env.operation_progression[machine]]-1)
                else:
                    action.append(self.solution[machine][self.env.operation_progression[machine]-1]-1)


            state, reward, done, _ = self.env.step(action)

            if done:

                print(f'\n\n __Environment Validation__\n'
                      f'Instance: {self.instance_id}, Makespan:{self.env.makespan}, Timestep:{self.env.time_step}'
                      f'\n'
                      f'____________________________________________________________________________\n\n')

        self.env.reset()
        self.env.close()

