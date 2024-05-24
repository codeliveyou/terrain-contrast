

def findMissingPieces(data, leftTop, rightBottom, rowNumber, columnNumber):
    N1, E1 = leftTop
    N2, E2 = rightBottom
    deltaN = (N1 - N2) / rowNumber
    deltaE = (E2 - E1) / columnNumber
    flagDictionary = [[0 for __ in range(columnNumber)].copy() for _ in range(rowNumber)]
    for map in data:
        DN1, DE1 = map[0]
        DN2, DE2 = map[1]
        DN = (DN1 + DN2) / 2
        DE = (DE1 + DE2) / 2
        row = int((N1 - DN) / deltaN)
        col = int((DE - E1) / deltaE)
        flagDictionary[row][col] = 1

    missingData = []

    flagDictionary[0][0] = 1
    flagDictionary[1][1] = 1

    for i in range(rowNumber):
        for j in range(columnNumber):
            if flagDictionary[i][j] == 0:
                DN1, DE1 = N1 - (i + 1) * deltaN, E1 + i * deltaE
                DN2, DE2 = N1 - i * deltaN, E1 + (i + 1) * deltaE
                missingData.append([[DN1, DE1], [DN2, DE2]])

    return missingData

N1, E1 = 39, 123
N2, E2 = 38, 125

leftTop = [N1, E1]
rightBottom = [N2, E2]

data = [] # [i = 0..total, [[N_i_1, E_i_1], [N_i_2, E_i_2]]] : N_i_1, E_i_1 -> left top cell, N_i_2, E_i_2 -> right bottom cell

rowNumber = 300
columnNumber = 400

reult = findMissingPieces(data, leftTop, rightBottom, rowNumber, columnNumber)

