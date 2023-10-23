import numpy as np
import gym
import copy
import random
import copy
from Gauss_Jordan_elimination import *
from model_env import model_simulation
import time
from tools import *

def generation_stateSpace_byModel(ite, circuit, model, num_qubit):
    state_space = []  ## state_space
    for index in range(ite):
        num_ite = len(circuit)
        model = model_simulation(circuit, num_qubit) ##new model, new SF
        for i in range(num_ite):
            state_now = model.state
            if i == 0:
                model.step(state_now, 0)  #the first step 0th gate put in 1th block
                # print("trajectory", model.get_trajectory())
                state_space.append(model.get_trajectory())  ## trajectory put in state space
            else:
                state_now = model.state

                # print("ith:", i)
                # print("state", model.state)
                ## action_limint will return a index where the current can put in start
                not_allowed = action_limit(i, circuit, state_now)
                # print("not_allowed", not_allowed)
                start = not_allowed
                end = i
                ## random choice a block where the current gate can put in
                action = random.randint(start, end)
                # print("action", action)
                model.step(state_now, action)
                # print("trajectory", model.get_trajectory()[3])
                state_space.append(model.get_trajectory()) ## trajectory put in state space
    return state_space

def generation_stateSpace_all(circuit, model):
    state_space = {}
    num_ite = len(circuit)
    for index in range(num_ite):
        if index == 0:
            state_space["{}th_layer".format(index)] = []
            state_now = model.state
            model.step(state_now, 0)

            trajectory = copy.deepcopy(model.get_trajectory())
            # print("0th", trajectory)
            state_space["{}th_layer".format(index)].append(trajectory)
            # print("0th_layer" ,state_space["0th_layer"])

        else:
            layer_upper = state_space["{}th_layer".format(index-1)]
            # print(f"{index-1}th layer_upper", layer_upper)
            state_space["{}th_layer".format(index)] = []
            # print(f"{index}th layer_now", state_space["{}th_layer".format(index)])

            for tra in layer_upper:
                # state_space["{}th_layer".format(index)].append([])
                state_input = copy.deepcopy(tra[3])
                # print("tra[3]_1", tra[3])
                # print("state_input1", state_input)
                not_allowed = action_limit(index, circuit, state_input)
                # print("not_allowed", not_allowed)
                start = not_allowed
                end = index + 1
                # print("start", start, "end", end)
                # print("tra[3]_2", tra[3])
                # print("state_input2", state_input)
                for ith_block in range(start, end):
                    state_input = copy.deepcopy(tra[3])
                    action =  ith_block
                    # print("action", action)
                    # print("state_input2", state_input)
                    model.step(state_input, action)

                    # print("tra[3]_3", tra[3])
                    # print("state_input3", state_input)
                    # print("adding ", model.get_trajectory())
                    state_space["{}th_layer".format(index)].append(model.get_trajectory())
                    # print(f"{index}th layer_now added", state_space["{}th_layer".format(index)])
    return state_space



class agent:
    def __init__(self, circuit, num_qubit,epsilon):
        self.circuit = circuit
        self.num_qubit = num_qubit
        self.state_space = None
        self.model = None
        self.table = None
        self.state = copy.deepcopy([circuit])

        # self.epsilon = epsilon

    def initialization(self):
        self.model = model_simulation(self.circuit, self.num_qubit)
        state_space = self.generation_state_space(0)
        self.initialization_table()

        # print("initialization: state_space", self.state_space)
        # print("initialization: self.q_table", self.table)

        ##底层赋值
        ith = str(len(self.state_space)-1)+"th_layer"
        last_layer = self.state_space[ith]
        # print("length:", len(self.state_space))
        # print("values", self.state_space.values())
        # print("last_layer：", last_layer)
        list_to_judge = []
        for item in last_layer:
            i = str(item[0])
            j = item[1]
            self.table[i][j] = item[2]
            list_to_judge.append(item[2])



        print("list_to_judge", list_to_judge)
        print("是否可优化：", any(num > 0 for num in list_to_judge))

        # print("updataed: self.q_table", self.table)

        # self.update_table()










    def initialization_table(self):
        # 初始化Q表
        Q_table = {}
        # 遍历state_space中的每一个层
        for layer in self.state_space.values():
            for state in layer:
                # print("state_all", state)
                s = str(state[0])  # 转换为tuple以使其可哈希
                a = state[1]
                # a = str(state[1])
                # print("state", s)
                # print("action", a)
                if s not in Q_table:
                    Q_table[s] = {}
                Q_table[s][a] = 0

        print("Q_table", Q_table)
        self.table = Q_table

    def preparation_model(self):
        self.model = model_simulation(self.circuit, self.num_qubit)
        return None

    def generation_state_space(self, ite):
        # model_start = self.preparation_model(self.circuit)
        if len(self.circuit) > 20:
            self.state_space = generation_stateSpace_byModel(ite, self.circuit, self.model)
        else:
            self.state_space = generation_stateSpace_all(self.circuit, self.model)

    def update_table(self, state):
        ##存在问题
        gamma = 1
        target = state
        states_to_next = [] ## 要更新的state的下一层
        for layer, lists in self.state_space.items():
            for l in lists:
                if l[0] == target:
                    states_to_next.append(l)
        # print("states_to_next", states_to_next)
        actions = [] ## 当前state有几种action可以选
        rewards = [] ## 当前state的action对应的reward
        state_next = [] ## 当前state用action后能到的下一个state
        q_valus_next = [] ##下一个state中的最大q值（？？？ 是不是也要乘epsilon）

        for item in states_to_next:
            actions.append(item[1])
            rewards.append(item[2])
            state_next.append(item[3])
        for item_next in state_next:
            # print("self.table[str(item_next)]", self.table[str(item_next)])
            # print("self.table[str(item_next)].values()", self.table[str(item_next)].values())
            max_value = max(self.table[str(item_next)].values())
            q_valus_next.append(max_value)

        ##当前state的 q值：reward + gamma*max(q值下一层)
        q_value_actions = [a + b * gamma for a, b in zip(rewards, q_valus_next)]
        for ith in range(len(actions)):
            self.table[str(state)][ith] = q_value_actions[ith]

    def search(self,state, action):
        # print("self.state_space", self.state_space)
        for layer in self.state_space:
                for elem in self.state_space[layer]:
                    # print("elem", elem)
                    # print("state", state)
                    # 检查元素的前两部分是否与给定的部分匹配
                    if elem[0] == state and elem[1] == action:
                        return elem


    def policy_function(self, state, epsilon):
        ## input state
        ## basing on q_table and following ε-greedy decide action
        # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
        #     f_a.write("in policy function" + '\n')
        #     f_a.write(f"epsilon:{epsilon}" + '\n')
        #     # f_a.writelines(str(agent.table) + '\n')
        #     f_a.write(f"state:{state}"+ '\n')
        target = state
        action = -1
        v_table = self.table
        v_table_single = v_table[str(target)]
        # print("v_table[str(target)]", v_table[str(target)])
        # print("v_table_single", v_table_single)
        # print("v_table_single", v_table_single)

        # 直接选择最优解
        # action = max(v_table_single, key=v_table_single.get)
        # print("action", action)


        #ε-greedy 基于最新更新的q-table 决定action
        if random.uniform(0, 1) > epsilon: ## 意外
            action = random.choice(list(v_table_single.keys()))
            # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
            #     f_a.write("意外解" + '\n')
            #     f_a.write(f"action:{action}" + '\n')
            # print("v_table_single", v_table_single)
            # print("list(v_table_single.keys())", list(v_table_single.keys()))
        else: ## 正常情况 选大的

            # print("v_table_single", v_table_single)
            # print("max(v_table_single, key=v_table_single.get)", max(v_table_single, key=v_table_single.get))
            action = max(v_table_single, key=v_table_single.get)
            # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
            #     f_a.write("最优解" + '\n')
            #     f_a.write(f"action:{action}" + '\n')
        # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
        #     f_a.write("policy function finsih" + '\n')
        return action

    def value_fucntion(self, state, action, epsilon):
        # print(".......................value function start....................")
        ##根据state, action计算value写在 q_table上
        ## v = r（basing state and action）+ v_next(v in state_next)
        ##state_next is from state and act action to get
        gamma = 1


        # print("input state", state, "input action", action)
        inform = self.search(state, action)
        # print("search inform", inform)
        reward = inform[2]
        # print("reward", reward)
        state_next = inform[3]
        # print("state_next", state_next)
        if state_next[-1] == []:
            # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
            #     f_a.write("finalstep"+f"location:{state}and{action}//"+f"reward:{reward}//" + '\n')
            self.table[str(state)][action] = reward

        else:
            # print("self.table[str(state_next)]", self.table[str(state_next)])
            value_next_max = max(self.table[str(state_next)].values())
            # print("value_next_max", value_next_max )



            #
            # action_next = self.policy_function(state_next, epsilon)
            # value_next = self.table[str(state_next)][action_next]
            # print("value_next", value_next)
            self.table[str(state)][action] = reward + value_next_max * gamma

            # self.table[str(state)][action] = reward + value_next_max * gamma
        #     with open("analyse.txt", 'a', encoding='utf-8') as f_a:
        #         f_a.write(f"location:{state}and{action}//" + f"reward:{reward}//" + f"value_next:{value_next}" '\n')
        # # print("v_table", self.table)
        # print(".......................value function end....................")

    def run(self, state,action):
        # state = self.state
        # state_space = self.state_space
        self.state = self.search(state, action)
        return self.state[3]







if __name__ == '__main__':
    ## 主程序 run

    circuit = [[0, 1], [4, 1], [5, 3], [4, 0], [1, 3], [4, 2], [2, 5], [2, 1], [4, 3], [0, 5]]

    thresold = 0.0001

    with open("result_6_10.txt", 'a', encoding='utf-8') as f:
        f.write(str(circuit) + '\n')

    agent = agent(circuit, 6, 0.7)
    agent.initialization()
    # print("agent.table", agent.table)
    start_time = time.time()
    episode = 100
    i = 0
    epsilon = 0.5
    decay_rate = 1/episode

    stop = 0
    path = []
    while i < episode:
        # print("num_episode", i)

        steps = len(circuit)
        state = [circuit]

        if i < episode/2:
            epsilon += (1 - epsilon) * decay_rate
        else:
            epsilon = 1
        # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
        #     f_a.write(f"{i}th" + '\n')
        #     f_a.write(f"epsilon:{epsilon}" + '\n')

        path = []
        pointer_stop = 0
        for ith_step in range(steps):

            # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
            #     f_a.writelines(str(agent.table) + '\n')
            #     f_a.write(f"state:{state}"+ '\n')
            # print("state_start", state)
            ## basing now state to determine action
            action = agent.policy_function(state, epsilon)
            # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
            #     f_a.write(f"action:{action}" + '\n')
            # # print("state:", agent.state, "action:", action)
            # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
            #     f_a.writelines("in value function" + '\n')
            v_origin = agent.table[str(state)][action]

            agent.value_fucntion(state, action, epsilon)

            v_updated = agent.table[str(state)][action]

            deta_v = abs(v_updated-v_origin)
            if deta_v < thresold:
                pointer_stop += 1


            # with open("analyse.txt", 'a', encoding='utf-8') as f_a:
            #     f_a.writelines("out value function" + '\n')
            #     f_a.writelines(str(agent.table) + '\n')

            state = agent.run(state, action)
            path.append(state)
            # print("path:", path)
            # print("state_end", state)

        if pointer_stop == steps:
            stop += 1



        # if
        ## result for one episode
        # print("path:", path)
        # print("state_space", agent.state_space)
        # print("v_table", agent.table)

        # if i == episode-1:
        #     with open(".record2.txt", 'a', encoding='utf-8') as f:
        #         f.write(str(path[-1]) + '\n')
        #         f.write(str(path) + '\n')
        #         f.writelines(str(agent.table) + '\n')
        #         f.writelines(str(agent.state_space) + '\n')
        i += 1



    end_time = time.time()
    elapsed_time = end_time - start_time
    # print("state_space", agent.state_space)
    with open("result_6_10.txt", 'a', encoding='utf-8') as f:

        # f.write(str(path) + '\n')
        # f.writelines(str(agent.table) + '\n')
        # f.writelines(str(agent.state_space) + '\n')
        f.write(str(f"程序运行时间：{elapsed_time}秒")+ '\n')
        # f.write(path)
        f.write(str(path[-1]) + '\n')
    print(f"程序运行时间：{elapsed_time}秒")

    ## 输出结果
    circuit = circuit
    circuits = path[-1]

    matrix = generate_matrix(circuit, 6)
    for line in matrix:
        print(line)

    cirres = gauss_jordan_elimination(matrix)
    matrix = cirres[0]
    for line in matrix:
        print(line)

    print("........................")

    sum = 0

    for item in circuits:
        matrix = generate_matrix(item, 6)
        result = gauss_jordan_elimination(matrix)
        matrix = result[0]
        for line in matrix:
            print(line)
        print("...................")
        print("result1", result[1])
        sum += result[1]
    print("initial:", cirres[1])
    print("divided:", sum)

# print("result.................")
    # print(circuit)
    # circuits = str(path[-1])
    # print(circuits)
    #
    # matrix = generate_matrix(circuit,4)
    # # for line in matrix:
    # #     print(line)
    #
    # cirres = gauss_jordan_elimination(matrix)
    # matrix = cirres[0]
    # # for line in matrix:
    # #     print(line)
    #
    # print("........................")
    #
    # sum = 0
    #
    # for item in circuits:
    #     matrix = generate_matrix(item,4)
    #     result = gauss_jordan_elimination(matrix)
    #     matrix = result[0]
    #     # for line in matrix:
    #     #     print(line)
    #     # print("...................")
    #     # print("result1", result[1])
    #     sum += result[1]
    # print("initial:", cirres[1])
    # print("divided:", sum)
    #
    #
    #
    #
    #
    #
