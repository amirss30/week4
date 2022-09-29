#1
A = []
N = int(raw_input("Input matrix size: "))
for i in range(0, N):
    row = []
    for j in range(0, N):
        row.append(int(raw_input("Input elements: ")))
    A.append(row)
sum = 0
for i in range(0, N):
    sum += A[i][i]
print sum
sum2 = 0
for i in range(0, N):
    for j in range(i+1, N):
        sum2 += A[i][j]
print sum2
#1(2)
def smallestInRow(mat, n, m):
    print("{", end="")
    for i in range(n):
        minm = mat[i][0]
        for j in range(1, m, 1):
            if (mat[i][j] < minm):
                minm = mat[i][j]
        print(minm, end=",")
    print("}")
def smallestInCol(mat, n, m):
    print("{", end="")
    for i in range(m):
        minm = mat[0][i]
        for j in range(1, n, 1):
            if (mat[j][i] < minm):
                minm = mat[j][i]
        print(minm, end=",")
    print("}"
if __name__ == '__main__':
    n = 3
    m = 3
    mat = [[2, 1, 7],
           [3, 7, 2],
           [5, 4, 9]];
    print("Minimum element of each row is",
          end=" ")
    smallestInRow(mat, n, m)
    print("Minimum element of each column is",
          end=" ")
    smallestInCol(mat, n, m)
    #2
    def isMagicSquare(mat):
        n = len(mat)
        sumd1 = 0
        sumd2 = 0
        for i in range(n):
            sumd1 += mat[i][i]
            sumd2 += mat[i][n - i - 1]
        if not (sumd1 == sumd2):
            return False
        for i in range(n):
            sumr = 0
            sumc = 0
            for j in range(n):
                sumr += mat[i][j]
                sumc += mat[j][i]
            if not (sumr == sumc == sumd1):
                return False
        return True
    mat = [[2, 7, 6],
           [9, 5, 1],
           [4, 3, 8]]
    if (isMagicSquare(mat)):
        print("Magic Square")
    else:
        print("Not a magic Square")
        #3
        def transpose(mat, tr, N):
            for i in range(N):
                for j in range(N):
                    tr[i][j] = mat[j][i]
        def isSymmetric(mat, N):
            tr = [[0 for j in range(len(mat[0]))] for i in range(len(mat))]
            transpose(mat, tr, N)
            for i in range(N):
                for j in range(N):
                    if (mat[i][j] != tr[i][j]):
                        return False
            return True
        mat = [[1, 3, 5], [3, 2, 4], [5, 4, 1]]
        if (isSymmetric(mat, 3)):
            print
            "Yes"
        else:
            print
            "No"
            #3(2)
            def search(mat, n, x):
                if (n == 0):
                    return -1
                for i in range(n):
                    for j in range(n):
                        if (mat[i][j] == x):
                            print("Element found at (", i, ",", j, ")")
                            return 1
                print(" Element not found")
                return 0
            if __name__ == "__main__":
                mat = [[10, 20, 30, 40], [15, 25, 35, 45],
                       [27, 29, 37, 48], [32, 33, 39, 50]]
                search(mat, 4, 29)
#4
def row_sum(arr):
    sum = 0
    print("\nFinding Sum of each row:\n")
    for i in range(m):
        for j in range(n):
            sum += arr[i][j]
        print("Sum of the row", i, "=", sum)
        sum = 0
def column_sum(arr):
    sum = 0
    print("\nFinding Sum of each column:\n")
    for i in range(m):
        for j in range(n):
            # Add the element
            sum += arr[j][i]
        print("Sum of the column", i, "=", sum)
        sum = 0
if __name__ == "__main__":
    arr = np.zeros((4, 4))
    x = 1
    for i in range(m):
        for j in range(n):
            arr[i][j] = x
            x += 1
    row_sum(arr)
    column_sum(arr)
    #4(2)
    def lower(matrix, row, col):
        for i in range(0, row):
            for j in range(0, col):
                if (i < j):
                    print("0", end=" ");
                else:
                    print(matrix[i][j],
                          end=" ");
            print(" ");
    def upper(matrix, row, col):
        for i in range(0, row):
            for j in range(0, col):
                if (i > j):
                    print("0", end=" ");
                else:
                    print(matrix[i][j],
                          end=" ");
            print(" ");
    matrix = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]];
    row = 3;
    col = 3;
    print("Lower triangular matrix: ");
    lower(matrix, row, col);
    print("Upper triangular matrix: ");
    upper(matrix, row, col);
#5
MAX_SIZE = 10
def sortByRow(mat, n, ascending):
    for i in range(n):
        if (ascending):
            mat[i].sort()
        else:
            mat[i].sort(reverse=True)
def transpose(mat, n):
    for i in range(n):
        for j in range(i + 1, n):
            temp = mat[i][j]
            mat[i][j] = mat[j][i]
            mat[j][i] = temp
def sortMatRowAndColWise(mat, n):
    sortByRow(mat, n, True)
    transpose(mat, n)
    sortByRow(mat, n, False)
    transpose(mat, n)
def printMat(mat, n):
    for i in range(n):
        for j in range(n):
            print(mat[i][j], " ", end="")
        print()
n = 3
mat = [[3, 2, 1],
       [9, 8, 7],
       [6, 5, 4]]
print("Original Matrix:")
printMat(mat, n)
sortMatRowAndColWise(mat, n)
print("Matrix After Sorting:")
printMat(mat, n)
#6
def maxelement(arr):
    # get number of rows and columns
    no_of_rows = len(arr)
    no_of_column = len(arr[0])
    for i in range(no_of_rows):
        max1 = 0
        for j in range(no_of_column):
            if arr[i][j] > max1:
                max1 = arr[i][j]
        print(max1)
arr = [[3, 4, 1, 8],
       [1, 4, 9, 11],
       [76, 34, 21, 1],
       [2, 1, 4, 5]]
maxelement(arr)
#8(2)
N = 4
def transpose(A, B):
    for i in range(N):
        for j in range(N):
            B[i][j] = A[j][i]
if __name__ == '__main__':
    A = [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]
    B = [[0 for x in range(N)] for y in range(N)]
    transpose(A, B)
    print("Result matrix is")
    for i in range(N):
        for j in range(N):
            print(B[i][j], " ", end='')
        print()
#9
MAX = 100
def largestSquare(matrix, R, C, q_i, q_j, K, Q):
    # Loop to solve for each query
    for q in range(Q):
        i = q_i[q]
        j = q_j[q]
        min_dist = min(min(i, j),
                       min(R - i - 1, C - j - 1))
        ans = -1
        for k in range(min_dist + 1):
            count = 0
            for row in range(i - k, i + k + 1):
                for col in range(j - k, j + k + 1):
                    count += matrix[row][col]
            if count > K:
                break
            ans = 2 * k + 1
        print(ans)
matrix = [[1, 0, 1, 0, 0],
          [1, 0, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [1, 0, 0, 1, 0]]
K = 9
Q = 1
q_i = [1]
q_j = [2]
largestSquare(matrix, 4, 5, q_i, q_j, K, Q)
#9(2)
MAX = 100
def binarySearch(mat, i, j_low,
                 j_high, x):
    while (j_low <= j_high):
        j_mid = (j_low + j_high) // 2
        if (mat[i][j_mid] == x):
            print("Found at (", i, ", ", j_mid, ")")
            return
        elif (mat[i][j_mid] > x):
            j_high = j_mid - 1
        else:
            j_low = j_mid + 1
    print("Element no found")
def sortedMatrixSearch(mat, n, m, x):
    # Single row matrix
    if (n == 1):
        binarySearch(mat, 0, 0, m - 1, x)
        return
    i_low = 0
    i_high = n - 1
    j_mid = m // 2
    while ((i_low + 1) < i_high):
        i_mid = (i_low + i_high) // 2
        if (mat[i_mid][j_mid] == x):
            print("Found at (", i_mid, ", ", j_mid, ")")
            return
        elif (mat[i_mid][j_mid] > x):
            i_high = i_mid
        else:
            i_low = i_mid
    if (mat[i_low][j_mid] == x):
        print("Found at (", i_low, ",", j_mid, ")")
    elif (mat[i_low + 1][j_mid] == x):
        print("Found at (", (i_low + 1), ", ", j_mid, ")")
    elif (x <= mat[i_low][j_mid - 1]):
        binarySearch(mat, i_low, 0, j_mid - 1, x)
    elif (x >= mat[i_low][j_mid + 1] and
          x <= mat[i_low][m - 1]):
        binarySearch(mat, i_low, j_mid + 1, m - 1, x)
    elif (x <= mat[i_low + 1][j_mid - 1]):
        binarySearch(mat, i_low + 1, 0, j_mid - 1, x)
else:
binarySearch(mat, i_low + 1, j_mid + 1, m - 1, x)
if __name__ == "__main__":
    n = 4
    m = 5
    x = 8
    mat = [[0, 6, 8, 9, 11],
           [20, 22, 28, 29, 31],
           [36, 38, 50, 61, 63],
           [64, 66, 100, 122, 128]]
    sortedMatrixSearch(mat, n, m, x)
    #11
    def findMinOpeartion(matrix, n):
        sumRow = [0] * n
        sumCol = [0] * n
        for i in range(n):
            for j in range(n):
                sumRow[i] += matrix[i][j]
                sumCol[j] += matrix[i][j]
        maxSum = 0
        for i in range(n):
            maxSum = max(maxSum, sumRow[i])
            maxSum = max(maxSum, sumCol[i])
        count = 0
        i = 0
        j = 0
        while i < n and j < n:
            diff = min(maxSum - sumRow[i],
                       maxSum - sumCol[j])
            matrix[i][j] += diff
            sumRow[i] += diff
            sumCol[j] += diff
            count += diff
            if (sumRow[i] == maxSum):
                i += 1
            if (sumCol[j] == maxSum):
                j += 1
        return count
    def printMatrix(matrix, n):
        for i in range(n):
            for j in range(n):
                print(matrix[i][j], end=" ")
            print()
    if __name__ == "__main__":
        matrix = [[1, 2],
                  [3, 4]]
        print(findMinOpeartion(matrix, 2))
        printMatrix(matrix, 2)
#12
def find(arr):
    n = len(arr)
    i = 0
    j = n - 1
    res = -1
    while i < n and j >= 0:
        if arr[i][j] == 0:
            while j >= 0 and (arr[i][j] == 0 or i == j):
                j -= 1
            if j == -1:
                res = i
                break
            else:
                i += 1
        else:
            while i < n and (arr[i][j] == 1 or i == j):
                i += 1
            if i == n:
                res = j
                break
            else:
                j -= 1
    if res == -1:
        return res
    for i in range(0, n):
        if res != i and arr[i][res] != 1:
            return -1
    for j in range(0, j):
        if res != j and arr[res][j] != 0:
            return -1;
    return res;
arr = [[0, 0, 1, 1, 0],
       [0, 0, 0, 1, 0],
       [1, 1, 1, 1, 0],
       [0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1]]
print(find(arr))
#13
def diagonalsMinMax(mat):
    n = len(mat)
    if (n == 0):
        return
    principalMin = mat[0][0]
    principalMax = mat[0][0]
    secondaryMin = mat[0][n - 1]
    secondaryMax = mat[0][n - 1]
    for i in range(1, n):
        for j in range(1, n):
            if (i == j):
                if (mat[i][j] < principalMin):
                    principalMin = mat[i][j]
                if (mat[i][j] > principalMax):
                    principalMax = mat[i][j]
            if ((i + j) == (n - 1)):
                if (mat[i][j] < secondaryMin):
                    secondaryMin = mat[i][j]
                if (mat[i][j] > secondaryMax):
                    secondaryMax = mat[i][j]
    print("Principal Diagonal Smallest Element: ",
          principalMin)
    print("Principal Diagonal Greatest Element : ",
          principalMax)

    print("Secondary Diagonal Smallest Element: ",
          secondaryMin)
    print("Secondary Diagonal Greatest Element: ",
          secondaryMax)
matrix = [[1, 2, 3, 4, -10],
          [5, 6, 7, 8, 6],
          [1, 2, 11, 3, 4],
          [5, 6, 70, 5, 8],
          [4, 9, 7, 1, -5]]
diagonalsMinMax(matrix)
#14
def maxDiagonalSum(arr, N):
    maxSum = 0
    for i in range(N):
        row = 0
        col = i
        sum1 = 0
        sum2 = 0
        while col < N and row < N:
            sum1 += arr[row][col]
            sum2 += arr[col][row]
            row += 1
            col += 1
        maxSum = max([sum1, maxSum, sum2])
    return maxSum
if __name__ == '__main__':
    mat = [[1, 2, 5, 7],
           [2, 6, 7, 3],
           [12, 3, 2, 4],
           [3, 6, 9, 4]]
    N = len(mat)
    print(maxDiagonalSum(mat, N))
    #15
    import sys
    N = 5
    def colMaxSum(mat):
        idx = -1
        maxSum = -sys.maxsize
        for i in range(0, N):
            for j in range(0, N):
                sum += mat[i][j]
            if (sum > maxSum):
                maxSum = sum
                idx = i
        res = [idx, maxSum]
        return res
    mat = [[1, 2, 3, 4, 5],
           [5, 3, 1, 4, 2],
           [5, 6, 7, 8, 9],
           [0, 6, 3, 4, 12],
           [9, 7, 12, 4, 3]]
    ans = colMaxSum(mat)
    print("Row", ans[0] + 1, "has max sum", ans[1])









