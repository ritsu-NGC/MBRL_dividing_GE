import copy


def generate_matrix(circuit,n):
    matrix = []
    input =  [[i] for i in range(n)]
    # print("input", input)
    for cnot in circuit:
        con = cnot[0]
        tar = cnot[1]
        tempo = copy.deepcopy(input[tar])
        input[tar] = list(set(input[con]).symmetric_difference(set(tempo)))

    for line_input in input:
        line = [0] * n
        for ele in line_input:
            line[ele] = 1
        matrix.append(line)

    return matrix





    # output = list(range(n))
    # # print("output", output)
    # for cnot in circuit:

    # return



def xor_rows(row1, row2):
    return [a ^ b for a, b in zip(row1, row2)]
def get_column(matrix, index_col):
    num_rows = len(matrix)
    values_in_column = []
    for index_row in range(index_col, num_rows):
        values_in_column.append(matrix[index_row][index_col])
    return values_in_column
def actions(column):
    operations = []
    go_next = False
    values_in_column = column
    value_in_diagonal = column[0]
    # print("value_in_diagonal:", value_in_diagonal)
    values_below_diagonal = column[1:]
    if all(x == 0 for x in values_below_diagonal):
        go_next = True

    if go_next == False:

        if value_in_diagonal == 0:
            # print("value_in_diagonal为零", value_in_diagonal)
            min_index = values_below_diagonal.index(1) + 1
            # print("min_index", min_index)
            while min_index > 0:
                operations.append([min_index,min_index-1])
                min_index -= 1
        else:
            indices_sorted = [index for index, value in enumerate(values_in_column) if value == 1]
            start, end = sorted(indices_sorted)[-2:]
            # print("start:", start)
            # print("end:", end)

            while start < end:
                operations.append([start,start+1])
                start += 1

    return go_next, value_in_diagonal, operations

def fold_matrix_diagonally(matrix):
    # 获取矩阵的行数和列数
    rows = len(matrix)
    cols = len(matrix[0])

    # 遍历矩阵的上三角
    for i in range(rows):
        for j in range(i+1, cols):
            # 交换元素
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    return matrix

def matrix_rever(matrix):
    for row in matrix:
        row.reverse()
    matrix.reverse()
    # for it in matrix:
        # print(it)
    return matrix
def gauss_jordan_elimination_down(matrix):
    sum_nna = 0

    num_rows = len(matrix)
    # print("行数：", num_rows)
    num_cols = len(matrix[0])
    # print("列数：", num_cols)

    for index_col in range(num_cols):
        i = 0
        # print(f"第{i}th")
        values_in_column = get_column(matrix, index_col)
        # print(f"第{index_col+1}列的所有元素", values_in_column)
        go_next = actions(values_in_column)[0]
        # print("go_next", go_next)
        operations = actions(values_in_column)[2]
        # print("operations", operations)

        while not go_next:
            i += 1
            # print(f"第{i}th")
            if i > 10:
                break
            ## 根据operations 执行更新矩阵

            for operation in operations:
                row1 = operation[0] + index_col
                row2 = operation[1] + index_col
                # print("operation", row1, row2)
                sum_nna += 1
                matrix[row2] = xor_rows(matrix[row1], matrix[row2])

                # for row in matrix:
                #     print(row)
                # print(".........................................")

            values_in_column = get_column(matrix, index_col)
            # print(f"第{index_col + 1}列的所有元素", values_in_column)
            go_next = actions(values_in_column)[0]
            # print("go_next", go_next)
            operations = actions(values_in_column)[2]
            # print("operations", operations)

    return matrix, sum_nna

def gauss_jordan_elimination_up(matrix):

    matrix = matrix_rever(matrix)
    result = gauss_jordan_elimination_down(matrix)
    return result

def gauss_jordan_elimination(matrix):
    result = gauss_jordan_elimination_down(matrix)
    sum_nna = result[1]
    # print("sum_nna",sum_nna)
    matrix = result[0]

    # print("matrix", matrix)
    # print(matrix[0][0])
    eles_in_diagonal = []
    for i in range(len(matrix)):
        eles_in_diagonal.append(matrix[i][i])
    if 0 in eles_in_diagonal:
        pass
        print("该circuit无法用gauss elimination完成一次")

    matrix = matrix_rever(matrix)
    sum_nna += gauss_jordan_elimination_down(matrix)[1]
    result = gauss_jordan_elimination_down(matrix)[0]
    return result, sum_nna

if __name__ == '__main__':
    circuit = [[0, 1], [1, 2], [2, 3], [3, 0], [3, 1]]
    circuit1 = [[0, 1], [1, 2]]
    circuit2 =  [[2, 3], [3, 0], [3, 1]]

    matrix = generate_matrix(circuit, 4)
    for line in matrix:
        print(line)
    print("...............")
    matrix = generate_matrix(circuit1, 4)
    for line in matrix:
        print(line)
    print("...............")
    matrix = generate_matrix(circuit2, 4)
    for line in matrix:
        print(line)

    print("...............")


    res = gauss_jordan_elimination(matrix)
    for item in res[0]:
        print(item)
    print(res[1])