import time
from copy import deepcopy
from Gauss_Jordan_elimination import *
from tools import  action_limit
import random
class model_simulation:
    ## input original circuit to initialize
    def __init__(self,circuit, num_qubit):
        self.circuit_ori = circuit
        self.state = [self.circuit_ori]   #state
        self.reward = 0     #sum of reward
        self.trajectory = [] #[s_i, action, s_i+1, reward(each step)]
        self.num_qubit = num_qubit

    def step(self, state_input, action):
        ## input s_i, action
        ## output s_i+1, reward
        self.trajectory = []
        self.state = state_input
        initial = deepcopy(self.state)
        # print("initial:", initial, len(initial))
        if len(initial) == 1:
            self.trajectory.append(initial)
            self.trajectory.append(action)
            self.state.insert(-1, [])

            CNOT_waiting_dividing = self.state[-1]
            # print("CNOT_waiting_dividing:", CNOT_waiting_dividing)
            # select the cnot waiting for dividing
            CNOT_seleted =  CNOT_waiting_dividing[0]
            # print("CNOT_seleted", CNOT_seleted)
            # basing on action, the cnot waiting is putted in some subcircuit
            self.state[action].append(CNOT_seleted)
            # reduce the selected cnot from the waiting cnot list
            self.state[-1] = CNOT_waiting_dividing[1:]
            # generation s_i+1 of is over
            # add s_i+1 in trajectory
            reward = 0 ## first step reward = 0
            self.trajectory.append(reward)
            self.trajectory.append(deepcopy(self.state))


        else:
            sum_nna_ini = 0
            sum_nna_new = 0
            # print("......................step after one...........................................")

            self.trajectory.append(initial) # add s_i in trajectory
            self.trajectory.append(action)   # add action in trajectory
            # print("self.trajectory", self.trajectory)




            # according to action, s_i → s_i+1

            self.state.insert(-1, [])     # add new subcircuit
            CNOT_waiting_dividing = self.state[-1]
            # select the cnot waiting for dividing
            CNOT_seleted =  CNOT_waiting_dividing[0]

            # basing on action, the cnot waiting is putted in some subcircuit
            self.state[action].append(CNOT_seleted)
            # reduce the selected cnot from the waiting cnot list
            self.state[-1] = CNOT_waiting_dividing[1:]
            # generation s_i+1 of is over
            # add s_i+1 in trajectory

            ## caluculation of sum_nna_ini
            # reward = sum_nna_ini - sum_nna_new

            num_circuit =  len(self.circuit_ori)
            num_waiting = len(self.state[-1])
            num_divided = num_circuit - num_waiting
            circuit_divided = self.circuit_ori[:num_divided]
            # print("circuit_divided", circuit_divided)
            matrix = generate_matrix(circuit_divided, self.num_qubit)
            sum_nna_ini += gauss_jordan_elimination(matrix)[1]

            for item_n in self.state[:-1]:
                # reward = sum_nna_ini - sum_nna_new
                # print("item_n", item_n)
                matrix_n = generate_matrix(item_n, self.num_qubit)
                sum_nna_new += gauss_jordan_elimination(matrix_n)[1]
            # print("sum_nna_ini", sum_nna_ini)
            # print("sum_nna_new", sum_nna_new)
            reward = sum_nna_ini - sum_nna_new
            # when the dividing of circuit is finished, give a big reward
            if self.state[-1] == []:
                reward = reward*1000

            self.reward += reward # sum of reward
            self.trajectory.append(reward) # add reward in trajectory
            # print("self.state before add in", self.state)
            self.trajectory.append(deepcopy(self.state))
            # print("self.trajectory after add in", self.trajectory[3])

        return self.state, reward

    def refresh(self): ## restart
        self.state = [self.circuit_ori]
        self.reward = 0

    def get_trajectory(self): ## obatian trajectory of each step
        return self.trajectory

if __name__ == '__main__':

    # one iteration adding correction
    start_time = time.time()
    state_space = []
    # circuit = [[0, 1], [0, 2], [4, 2], [2, 3], [3, 4],]
    # num_ite =  len(circuit)
    # print("num_ite", num_ite)
    # model = model_simulation(circuit)
    # ites = 10
    # for index in range(ites):
    #     circuit = [[0, 1], [0, 2], [4, 2], [2, 3], [3, 4], ]
    #     num_ite = len(circuit)
    #     model = model_simulation(circuit)
    #     for i in range(num_ite):
    #
    #         state_now = model.state
    #         if i == 0:
    #             model.step(state_now, 0)
    #             # print("trajectory", model.get_trajectory())
    #             state_space.append(model.get_trajectory())
    #
    #         else:
    #             state_now = model.state
    #             print("ith:", i)
    #             print("state", model.state)
    #             not_allowed = action_limit(i, circuit, state_now)
    #             print("not_allowed", not_allowed)
    #             start = not_allowed
    #             end = i
    #             state_now = model.state
    #
    #             action = random.randint(start, end)
    #             # print("action", action)
    #             model.step(state_now, action)
    #             # print("trajectory", model.get_trajectory()[3])
    #             state_space.append(model.get_trajectory())
    #
    # print("///////////////////////////////////")
    # print(len(state_space))
    # for state in state_space:
    #     print("state_space", state)

    circuit = [[0, 1], [0, 2], [4, 2], [2, 3], [3, 4], ]
    model = model_simulation(circuit)
    model.step([[[0, 1], [0, 2], [4, 2]], [], [], [[2, 3], [3, 4]]], 2)
    print(model.get_trajectory())




    end_time = time.time()
    elapsed_time = end_time - start_time
