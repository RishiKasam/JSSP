import gym
import datetime
import numpy as np
import pandas as pd
from gym import spaces
import plotly.express as py
import imageio
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os.path


class Jssp(gym.Env):
    def __init__(self, instance_id=None, hyper_parameters=None, roll_out_timestep=None):
        self.instance_matrix = None  # contains machine order and respective time
        self.jobs = None
        self.machines = None
        self.reward = None
        self.time_step = 0
        self.job_detail = None
        self.job_instanceMatrix(instance_id=instance_id)  # to parse the instances
        self.roll_out = roll_out_timestep

        self.machine_temp_memory = None

        self.operationMatrix = None
        self.occupancy_average = None
        self.occupancy_history = None
        self.occupancy_current_average = None
        self.next_time_step = None
        self.makespan = None
        self.machine_available_jobs = None
        self.machine_available_jobs_time = None
        self.job_current_machine = None
        self.machine_current_job = None
        self.machine_assignment_step = None
        self.job_needs_machine = None
        self.last_time_step = None
        self.machine_index = None
        self.hyper_parameters = hyper_parameters
        self.agent_steps = None
        self.jobs_need_machines = None
        self.total_reward = None


        assert self.jobs is not None
        assert self.machines is not None
        assert self.instance_matrix is not None
        '''
        gives the progression status of each machine and job
        This progresssion is used to obtain the next action from the agent
        '''
        self.job_progression = None  # job progression
        self.operation_progression = None  # machine progression

        '''
        these parameters indicate the current status of each machine and jobs
        jobs and machines are allocated based on these parameters
        '''
        self.machines_status = np.zeros(self.machines,
                                        dtype=bool)  # status of the machine
        self.job_status = np.zeros(self.jobs + 1, dtype=bool)  # status of the jobs

        '''
        To create a realistic machine environment using simpy

        https://simpy.readthedocs.io/en/latest/examples/machine_shop.html
        '''

        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]

        self.job_process_time = np.zeros(self.jobs, dtype=int)  # provides the time
        self.job_process_status = np.zeros(self.jobs,
                                           dtype=int)  # machine number  carried out  in the particular machine
        self.machines_process_status = np.zeros(self.machines,
                                                dtype=int)  # job number  carried out  in the particular machine

        self.machines_Left_over_time = np.zeros(self.machines,
                                                dtype=int)  # Time that is left for the machine to be free
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]


        '''
        multi-discrete action space
            the index of the list represents the job number 
            the value in the list represents the machine number 
        '''
        self.action_space = gym.spaces.MultiDiscrete(
            [self.jobs for _ in range(self.machines)])
        '''
        Observation space gives the machine progression
        '''

        observation_space = {'machine status': spaces.Box(low=0, high=1, shape=(self.machines,), dtype=bool),
                             'job status': spaces.Box(low=0, high=1, shape=(self.jobs,), dtype=bool),
                             'job progression': spaces.Box(low=0, high=self.machines, shape=(self.jobs,), dtype=int),
                             'machining progression': spaces.Box(low=0, high=self.jobs, shape=(self.machines,), dtype=int),
                             'total job status': spaces.Box(low=0, high=1, shape=(self.jobs, self.machines), dtype=bool),
                             'job availability': spaces.Box(low=0, high=1, shape=(self.machines, self.jobs), dtype=bool),
                             'available job times': spaces.Box(low=0, high=200, shape=(self.machines, self.jobs), dtype=int),

                             'machine time left' : spaces.Box(low=0, high=self.roll_out, shape=(self.machines,), dtype=int),
                             'job time left' : spaces.Box(low=0, high=self.roll_out, shape=(self.jobs,), dtype=int),
                             'occupancy average' : spaces.Discrete(100),
                             'machine current job' : spaces.Box(low=0, high=self.jobs, shape=(self.machines,), dtype=int),
                             'job current machine' : spaces.Box(low=0, high=self.machines, shape=(self.jobs,), dtype=int)


                             }

        if hyper_parameters is None:
            state = observation_space
            self.verbose = False
        else:
            self.verbose = False
            state = {}
            i = 0
            for key in hyper_parameters:
                if not key.__contains__('info'):
                    if i > 0 and hyper_parameters[key] != 0:
                        state[key] = observation_space[key]
                    i += 1

        self.observation_space = gym.spaces.Dict(state)
        #self.observation_space = gym.spaces.Dict(observation_space)

    def reset(self):
        self.job_detail = np.zeros(self.jobs, dtype=int)
        self.start_time = datetime.datetime.now().timestamp()
        self.operation_progression = np.zeros(self.machines, dtype=int)
        self.job_progression = np.zeros(self.jobs, dtype=int)
        self.machines_status = np.zeros(self.machines,
                                        dtype=bool)  # maximum value indicating the maximum number machines available
        self.job_status = np.zeros(self.jobs, dtype=bool)
        self.machines_process_status = np.zeros(self.machines,
                                                dtype=int)  # job number  carried out  in the particular machine
        self.machines_Left_over_time = np.zeros(self.machines, dtype=int)
        self.operationMatrix = np.zeros((self.jobs, self.machines), dtype=bool)
        self.reward = 0
        self.time_step = 0

        self.last_time_step = 0
        self.machine_current_job = np.full(self.machines, -1, dtype=int)
        self.job_current_machine = np.full(self.jobs, -1, dtype=int)
        self.job_process_status = np.zeros(self.jobs, dtype=int)
        self.next_time_step = 2000
        self.occupancy_average = 0.5
        self.occupancy_current_average = 0
        self.occupancy_history = []
        self.makespan = 10000
        self.total_reward = 0

        self.machine_available_jobs = np.zeros((self.machines, self.jobs), dtype=bool)
        self.machine_available_jobs_time = np.full((self.machines, self.jobs), -1, dtype=int)
        self.machine_temp_memory = np.ones(self.machines, dtype=bool)

        self.machine_assignment_step = np.zeros(self.machines, dtype=int)
        self.job_needs_machine = np.zeros(self.jobs, dtype=int)

        self.jobs_need_machines = np.zeros((self.jobs, self.machines), dtype=int)

        '''for machine in range(self.machines):

            for job in range(self.jobs):
                self.jobs_need_machines[machine][job] = self.instance_matrix[job][machine][0]
                self.machine_available_jobs[self.instance_matrix[job][machine][0]][job] = True
                self.machine_available_jobs_time[self.instance_matrix[job][machine][0]][job] = self.instance_matrix[job][machine][1]'''

        #job order update
        for job in range(self.jobs):
            self.job_needs_machine[job] = self.instance_matrix[job][0][0]
            self.machine_available_jobs[self.instance_matrix[job][0][0]][job] = True
            self.machine_available_jobs_time[self.instance_matrix[job][0][0]][job] = self.instance_matrix[job][0][1]

        #update temp memory
        
        for machine in range(self.machines):
            if not self.machine_available_jobs[machine][:].any():
                self.machine_temp_memory[machine] = False

        return self._update_state()

    '''
    job instance matrix consists of the job and its time details 
    '''

    def job_instanceMatrix(self,instance_id='la16'):
        with open(os.getcwd() + fr'\JsspEnvironment\envs\Instances\{instance_id}', 'r') as instance_file:
            line_str = instance_file.readline()
            line_cnt = 1
            while line_str:
                split_data = line_str.split()
                if line_cnt == 1:
                    self.jobs, self.machines = int(split_data[0]), int(split_data[1])
                    # matrix which store tuple of (machine, length of the job)
                    self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=(int, 2))
                    # contains all the time to complete jobs
                    self.jobs_length = np.zeros(self.jobs, dtype=int)
                else:
                    # couple (machine, time)
                    assert len(split_data) % 2 == 0
                    # each jobs must pass a number of operation equal to the number of machines
                    assert len(split_data) / 2 == self.machines
                    i = 0
                    # we get the actual jobs
                    job_nb = line_cnt - 2
                    while i < len(split_data):
                        machine, time = int(split_data[i]), int(split_data[i + 1])
                        self.instance_matrix[job_nb][i // 2] = (machine, time)

                        i += 2
                line_str = instance_file.readline()
                line_cnt += 1



    '''
    assigning the job to the requested machine(machines)
    '''

    def _assign_job(self, actions):
        for machine_nb in range(self.machines):
            job_nb = actions[machine_nb]

            if not actions[machine_nb] == self.jobs and not self.operationMatrix[actions[machine_nb]][machine_nb]:
                if self.job_needs_machine[actions[machine_nb]] == machine_nb:
                    if not self.job_status[actions[machine_nb]] and not self.machines_status[machine_nb]:

                        self.machines_status[machine_nb] = True
                        self.machine_assignment_step[machine_nb] = self.time_step
                        self.operationMatrix[actions[machine_nb]][machine_nb] = True #total_job_status[job_id][machine_id]
                        self.operation_progression[machine_nb] += 1

                        self.machine_current_job[machine_nb] = actions[machine_nb]
                        self.job_current_machine[actions[machine_nb]] = machine_nb
                        #self.machines_process_status[machine_nb] = actions[machine_nb]
                        #self.job_process_status[actions[machine_nb]] = machine_nb
                        self.machine_available_jobs[machine_nb][actions[machine_nb]] = False
                        self.machine_available_jobs_time[machine_nb][actions[machine_nb]] = -1

                        self.job_status[actions[machine_nb]] = True
                        self.job_progression[actions[machine_nb]] += 1
                        if self.job_progression[actions[machine_nb]] < self.machines:
                            self.job_needs_machine[actions[machine_nb]] = \
                                self.instance_matrix[actions[machine_nb]][self.job_progression[actions[machine_nb]]][0]
                        for machine in range(self.machines):
                            if machine_nb == self.instance_matrix[actions[machine_nb]][machine][0]:  # machine comparision to retreive the processing time
                                self.machines_Left_over_time[machine_nb] = \
                                    self.instance_matrix[actions[machine_nb]][machine][1] + self.time_step #machine_finish_time or processing time
                                self.job_process_time[actions[machine_nb]] = self.machines_Left_over_time[machine_nb] #job_left_over_time[machine_id]
                                self.job_detail[actions[machine_nb]] = self.time_step
                                """print(
                                    f'assigning machine: {machine_nb}, job: {actions[machine_nb]} '
                                    f'for {self.machines_Left_over_time[machine_nb]} at {self.time_step}\n')
                                    #f'with job_detail {self.job_detail[actions[machine_nb]]}"""

                        #self.machines_process_status[machine_nb] = actions[machine_nb]  #machine_current_job
                        #self.job_process_status[actions[machine_nb]] = machine_nb   #job_current_machine

                        if self.operationMatrix.all():
                            self.makespan = max(self.machines_Left_over_time)

                        assert self.machines_status[machine_nb] == True, 'Particular machine status should be changed'
                        assert self.job_status[actions[machine_nb]] == True, 'Particular job status should be changed'

            self.machine_temp_memory[machine_nb] = False

    def _update_state(self): #states to update after an interation controlled by the one hot encoded key

        observation_space = {'machine status': self.machines_status,
                             'job status': self.job_status,
                             'job progression': self.job_progression,
                             'machining progression': self.operation_progression,
                             'total job status': self.operationMatrix,
                             'job availability': self.machine_available_jobs,
                             'available job times': self.machine_available_jobs_time,
                             'machine time left': self.machines_Left_over_time,
                             'job time left' : self.job_process_time,
                             "occupancy average": int(self.occupancy_average * 100),
                             'machine current job' : self.machine_current_job,
                             'job current machine' : self.job_current_machine

                                 }
        if self.hyper_parameters is None:
            state = observation_space
        else:
            state = {}
            i = 0
            for key in self.hyper_parameters:
                if not key.__contains__('info'):
                    if i > 0 and self.hyper_parameters[key] != 0:
                        state[key] = observation_space[key]
                    i += 1
        self.observation_space = state
        return self.observation_space



    def step(self, actions):
        self._assign_job(actions)
        reward = self._reward_calculation()
        self._next_step_iter(actions)
        self.total_reward += reward
        done = self._check_done()
        state = self._update_state()

        return state, reward, done, {}

    def _next_step_iter(self, actions):
        """
        Time Step Transition
        :return:
        """
        self.last_time_step = self.time_step

        while np.where(self.machine_temp_memory)[0].__len__() == 0 and not self.operationMatrix.all():

            self._add_time_step()
            self._update_operation(actions)
            self._select_next_machines()
        if self.operationMatrix.all():
            self.time_step = max(self.machines_Left_over_time)
            self._update_operation(actions)

    def _select_next_machines(self): #selecting the next machine

        for machine_nb in range(self.machines):
            if not self.machines_status[machine_nb] and self.machine_available_jobs[machine_nb][:].any():
                self.machine_temp_memory[machine_nb] = True

    def _add_time_step(self): #choosing the next time step top jump to

        if any(self.machines_status):
            next_step = 1000
            for machine_nb in range(self.machines):
                if self.machines_status[machine_nb]:
                    next_step = min(next_step, self.machines_Left_over_time[machine_nb] - self.time_step)
                    self.next_time_step = next_step
        else:
            self.next_time_step = 1

        self.time_step += self.next_time_step
        self.occupancy_current_average = (np.count_nonzero(self.machines_status) / self.machines)
        self.occupancy_history.append(self.occupancy_current_average)
        self.occupancy_average = np.average(self.occupancy_history)


    def _update_operation(self, actions): # Time step is used to indicate scheduler as well as time
        for machine_nb in range(self.machines):
            if self.machines_status[machine_nb]:
                if self.machines_Left_over_time[machine_nb] <= self.time_step:
                    job_nb = self.machine_current_job[machine_nb]

                    self.job_status[job_nb] = False
                    self.machines_status[machine_nb] = False
                    self.machine_current_job[machine_nb] = -1
                    self.job_current_machine[job_nb] = -1

                    if self.job_progression[job_nb] < self.machines and job_nb != self.jobs:
                        machine_required = \
                            self.instance_matrix[job_nb][self.job_progression[job_nb]][0]
                        self.job_needs_machine[job_nb] = machine_required
                        self.machine_available_jobs[machine_required][job_nb] = True
                        self.machine_available_jobs_time[machine_required][job_nb] = self.instance_matrix[job_nb][0][1]


    def _reward_calculation(self):
        reward = 0
        machine_reward = 10
        for machine in range(self.machines):
            if self.machine_assignment_step[machine] == self.time_step and self.machines_status[machine]:
                reward += 1

        '''for job in range(self.jobs):
            if self.operationMatrix[job][:].all():
                reward += 5'''
        if self.operationMatrix.all():
            last_reward = self.roll_out - self.makespan
            if last_reward <= 0:
                last_reward = 1
            reward += last_reward * 1.5

        return reward

    def _check_done(self):
        if self.operationMatrix.all() == True or self.time_step >= self.roll_out:  # Time step indirectly has the makespan of the jobs
            print("============================================")
            print("time step", self.time_step)
            print("makespan", self.makespan)
            print("operation progression", self.operation_progression)
            print("total reward", self.total_reward)
            print("============================================")
            return True
        else:
            return False

    def render(self, mode="human"):

        # https://plotly.com/python/gantt/
        df = []
        for i in range(self.jobs):
            df.append(dict(Task=f'Job {i}', Start=datetime.datetime.fromtimestamp(self.job_detail[i]),
                           End=datetime.datetime.fromtimestamp(self.job_process_time[i]),
                           Resource=f'Machine {self.job_process_status[i]}'))

        df = pd.DataFrame(df)

        fig = py.timeline(df, x_start="Start", x_end="End", y="Task", color="Resource")
        # fig = ff.create_gantt(df, colors=self.colors, index_col='Resource', show_colorbar=True,
        # group_tasks=True)
        fig.update_yaxes(autorange="reversed")
        return fig
