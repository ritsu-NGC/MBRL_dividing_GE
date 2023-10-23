from Gauss_Jordan_elimination import *
from main import agent

if __name__ == '__main__':
    # agent1 = agent()
    num_qubit = 6

    circuit =[[2, 3], [4, 0], [4, 2], [0, 1], [5, 1], [5, 2], [1, 2], [3, 1], [1, 4]]
    circuits =[[[3, 4]], [[3, 0], [0, 2], [2, 3], [0, 1], [3, 1], [4, 2], [4, 1], [0, 4]], [], [], [], [], [], [], [], []]

    matrix = generate_matrix(circuit, num_qubit)
    for line in matrix:
        print(line)

    cirres = gauss_jordan_elimination(matrix)
    matrix = cirres[0]
    for line in matrix:
        print(line)

    print("........................")

    sum = 0

    for item in circuits:
        matrix = generate_matrix(item, num_qubit)
        result = gauss_jordan_elimination(matrix)
        matrix = result[0]
        for line in matrix:
            print(line)
        print("...................")
        print("result1", result[1])
        sum += result[1]
    print("initial:", cirres[1])
    print("divided:", sum)

    # circuits = [
    #     [[2, 3], [1, 0], [0, 2], [1, 3], [3, 0], [2, 1]],
    #     [[3, 0], [2, 3], [2, 1], [3, 1], [0, 2], [0, 1]],
    #     [[2, 0], [1, 2], [3, 1], [3, 2], [3, 0], [1, 0]],
    #     [[0, 2], [0, 1], [3, 2], [3, 1], [1, 2], [3, 0]],
    #     [[0, 2], [3, 0], [1, 0], [2, 1], [2, 3], [3, 1]],
    #     [[2, 3], [1, 3], [0, 2], [3, 0], [2, 1], [1, 0]],
    #     [[1, 0], [1, 2], [3, 0], [1, 3], [2, 3], [0, 2]],
    #     [[1, 2], [0, 1], [2, 3], [3, 1], [0, 2], [3, 0]],
    #     [[0, 1], [1, 3], [0, 2], [3, 2], [0, 3], [1, 2]],
    #     [[3, 2], [3, 0], [2, 1], [2, 0], [1, 0], [1, 3]],
    # ]
    # i = 0
    # for item in circuits:
    #     print(i)
    #     print(item)
    #     matrix = generate_matrix(item,4)
    #     result = gauss_jordan_elimination(matrix)
    #     matrix = result[0]
    #     for line in matrix:
    #         print(line)
    #     print("...................")
    #     i += 1
        # print("result1", result[1])
        # sum += result[1]
    # print("divided:", sum)









