import copy
import random
# from main import agent
## keep cnot circuit functionality
def blocks_correct(blocks,list,step):
    # print("blocks", blocks)

    cnots = copy.deepcopy(list)
    cnot_inspect = cnots[step]

    control = cnot_inspect[0]
    target = cnot_inspect[1]
    # print("con and tar", control, target)

    control_before = []
    targets_before = []

    for item in cnots[:step]:
        control_before.append(item[0])
        targets_before.append(item[1])
    # print("target bits before cnot_inspection", targets_before)
    block_noin = 0
    indexes_tar = [index for index, item in enumerate(targets_before) if item == control]
    indexes_con = [index for index, item in enumerate(control_before) if item == target]

    # print("indexes",indexes_tar)
    block_prev = []
    for i in indexes_tar:
        target_before = cnots[i]
        # print("target:",target)
        for index in range(len(blocks)):
            # print("index in correct",index)
            # print("blocks in correct",blocks)
            for item in blocks[index]:
                if item == target_before:
                    block_prev.append(int(index))
                else:
                    block_prev.append(-1)

    for i in indexes_con:
        control_before= cnots[i]
        for index in range(len(blocks)):
            for item in blocks[index]:
                if item == control_before:
                    block_prev.append(int(index))
                else:
                    block_prev.append(-1)

                # print("block_prev:",block_prev)
    if block_prev == []:
        block_prev_max = 0
    else:
        block_prev_max = max(block_prev)

    if block_noin < block_prev_max:
        block_noin = block_prev_max

    return block_noin
def remove_keys_by_index(d, idx):
    keys = list(d.keys())
    new_keys = keys[idx:]
    return {key: d[key] for key in new_keys}
## keep cnot circuit functionality



def generation_stateSpace(circuit):
    # should input state, now is single cnot
    cnots = copy.deepcopy(circuit)

    tree = TreeNode(0, 1, cnots[0])

    for i in range(1, len(cnots)):
        # print("i:", i)
        # get to know which block the cnot can not put in
        print("cnot_selected", cnots[i])
        print("cnots_preceding", cnots[:i])
        mutex_list = triming(cnots[i], cnots[:i])
        print("mutex_list", mutex_list)
        block_start = 0
        for item in mutex_list:
            print("mutex_cnots",cnots[item])
            nodes_mutex = tree.find_nodes(item)
            # print("mutex_cnots find nodes", nodes_mutex)

            # if block_start < tree.find_nodes(item):
            #     block_start =  tree.find_nodes(item)


        num_blocks = i + 1
        for block in range(1, num_blocks+1):

            # print(f"block{block}")

            if num_blocks > 1:
                matching_nodes = tree.find_nodes(i-1)
                print("matching_nodes", matching_nodes)
                for node_upper in matching_nodes:
                    # Create a new node for each node_upper
                    node = TreeNode(i, block, cnots[i])
                    node_upper.add_child(node)
            else:
                node = TreeNode(i, block, cnots[i])
                tree.add_child(node)

    return tree

## state space
##  [s_0_1, v [next layer] ]
##  [[s_1_1, v [next layer]], [s_1_2, v [next layer]]]
class TreeNode:
    def __init__(self, layer, block, state):
        self.layer = layer
        self.block = block
        self.state = state
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def remove_child(self, child_node):
        self.children.remove(child_node)

    def find_nodes(self, target_layer):
        nodes = []
        if self.layer == target_layer:
            nodes.append(self)

        for child in self.children:
            nodes.extend(child.find_nodes(target_layer))

        return nodes

    def __repr__(self, level=0):
        ret = "\t" * level + f"Layer: {self.layer}, Block: {self.block}, State: {self.state}" + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


def triming(cnot_selected, cnots_preceding):

    control = cnot_selected[0]
    target = cnot_selected[1]

    control_before = []
    targets_before = []
    for item in cnots_preceding:
        control_before.append(item[0])
        targets_before.append(item[1])
    # print("target bits before cnot_inspection", targets_before)
    mutex_list = []
    indexes_tar = [index for index, item in enumerate(targets_before) if item == control]
    indexes_con = [index for index, item in enumerate(control_before) if item == target]
    mutex_list = list(set(indexes_tar) | set(indexes_con))
    return mutex_list  ## return number of layer

def action_limit(i, cnots, state):
    cnot_selected = cnots[i]
    # print("cnot_selected", cnot_selected)
    cnots_preceding = cnots[:i]
    # print("cnots_preceding", cnots_preceding)
    circuit_divided = state
    # print("circuit_divided", circuit_divided)

    control = cnot_selected[0]
    target = cnot_selected[1]

    control_before = []
    targets_before = []
    for item in cnots_preceding:
        control_before.append(item[0])
        targets_before.append(item[1])
    # print("target bits before cnot_inspection", targets_before)
    indexes_tar = [index for index, item in enumerate(targets_before) if item == control]
    indexes_con = [index for index, item in enumerate(control_before) if item == target]
    mutex_list = list(set(indexes_tar) | set(indexes_con))

    block_limined = []
    for index in mutex_list:
        target = cnots[index]
        for i, sublist in enumerate(circuit_divided):
            if target in sublist:
                block_limined.append(i)
    # print("block_limined", block_limined)
    if block_limined == []:
        start = 0
    else:
        start = max(block_limined)
    return start



def Random_gates(qubit_num, gate_num):
    gate_list = []
    while len(gate_list) < gate_num:
        gate = random.sample(range(0, qubit_num), 2)
        mid = [gate[1], gate[0]]
        if gate not in gate_list and mid not in gate_list:
            gate_list.append(gate)


        # gate_list.append(gate)

    return gate_list


if __name__ == '__main__':
    num_circuit = 10
    circuits = []

    for i in range(num_circuit):
        list = Random_gates(6, 10)
        if list not in circuits:
            circuits.append(list)
            print(list)

    # agnet1 = agent()
    # for item in list:
        # agnet1 = agent(item, 5, 0)
        # agent1 = agent(item)
        # agent1.initialization()






