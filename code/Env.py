# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger
MAX_TIME = 24*30


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # (1,2) (1,3)
        self.action_space = [(0, 0)]+[(p, q) for p in range(m)
                                      for q in range(m) if p != q]
        self.state_space = [(xi, tj, dk) for xi in range(m)
                            for tj in range(t) for dk in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()

        self.action_space_size = len(self.action_space)
        self.total_time = 0

    # Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros((m + t + d))
        location, time, day = state
        state_encod[location] = 1
        state_encod[m + time] = 1
        state_encod[m + t + day] = 1
        return state_encod

    # Use this function if you are using architecture-2
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

    #     return state_encod

    # Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location, time, day = state
        poisson_map = {
            0: 2,
            1: 12,
            2: 4,
            3: 7,
            4: 8
        }
        requests = np.random.poisson(poisson_map[location])

        if requests > 15:
            requests = 15

        # (0,0) is not considered as customer request
        possible_actions_index = random.sample(
            range(1, self.action_space_size), requests)
        actions = [self.action_space[i] for i in possible_actions_index]

        # (0,0) is an allowed action which can be taken by the driver
        actions.append([0, 0])
        possible_actions_index.append(0)
        return possible_actions_index, actions

    def updated_time_and_day(self, time, day, ride_time):
        updated_time = (time+ride_time) % t
        updated_day = (day + (time+ride_time)//t) % d
        return updated_time, updated_day

    def updated_time_and_day_from_location(self, from_location, to_location, time, day, Time_matrix):
        ride_time = Time_matrix[from_location, to_location, time, day]
        updated_time, updated_day = updated_time_and_day(time, day, ride_time)
        return ride_time, updated_time, updated_day

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        # 𝑅𝑘∗(𝑇𝑖𝑚𝑒(𝑝,𝑞))−𝐶𝑓∗(𝑇𝑖𝑚𝑒(𝑝,𝑞)+𝑇𝑖𝑚𝑒(𝑖,𝑝))
        location, time, day = state
        pick_up, drop = action

        # driver did not take the request
        if pick_up == 0 and drop == 0:
            reward = -C
        else:
            if location != pick_up:
                # when reaching from current location to pickup, the time of the day and day of the week can change
                ride_time, updated_time, updated_day = self.updated_time_and_day_from_location(location, pick_up,
                                                                                               time, day, Time_matrix)

                reward = R*Time_matrix[pick_up, drop, updated_time, updated_day]-C*(
                    Time_matrix[pick_up, drop, updated_time, updated_day]+ride_time)
            else:
                reward = R*Time_matrix[pick_up, drop, time, day] - \
                    C*Time_matrix[pick_up, drop, time, day]

        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        location, time, day = state
        pick_up, drop = action

        if pick_up == 0 and drop == 0:
            self.total_time = self.total_time + 1
            next_time, next_day = self.updated_time_and_day(time, day, 1)
        else:
            ride_time_till_pick_up, updated_time, updated_day = self.updated_time_and_day_from_location(location, pick_up,
                                                                                                        time, day, Time_matrix)

            time_spent_after_pick_up = Time_matrix[pick_up,
                                                   drop, updated_time, updated_day]
            self.total_time = self.total_time + \
                ride_time_till_pick_up+time_spent_after_pick_up

            next_time, next_day = self.updated_time_and_day(
                updated_time, updated_day, time_spent_after_pick_up)

        is_finished = self.total_time < MAX_TIME
        if is_finished:
            self.total_time = 0

        return (drop, next_time, next_day), is_finished

    def reset(self):
        return self.action_space, self.state_space, self.state_init
