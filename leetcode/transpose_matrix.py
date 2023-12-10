import unittest

class TransposeMatrix():
    def run(self, matrix: list[list[int]]) -> list[list[int]]:
        if not matrix[0]:
            return []
        len_sub: int = len(matrix[0])
        len_new_sub: int = len(matrix)

        curr_new_sub: list[int] = []

        i: int = 0

        for i in range(len_new_sub):
            for k in range(len_sub):
                curr_new_sub.append()


class TestTransposeMatrix():
    def setUp(self):
        self.transpose_matrix = TransposeMatrix()

    def test_transpose_matrix(self):
        self.assertEqual(self.transpose_matrix.run([[1,2,3],[4,5,6],[7,8,9]]), [[1,4,7],[2,5,8],[3,6,9]])
        self.assertEqual(self.transpose_matrix.run([[1,2,3],[4,5,6]]), [[1,4],[2,5],[3,6]])

"""
867. Transpose Matrix
Easy
3.1K
430
Companies
Given a 2D integer array matrix, return the transpose of matrix.

The transpose of a matrix is the matrix flipped over its main diagonal, switching the matrix's row and column indices.



 

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[1,4,7],[2,5,8],[3,6,9]]
Example 2:

Input: matrix = [[1,2,3],[4,5,6]]
Output: [[1,4],[2,5],[3,6]]
 

Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 1000
1 <= m * n <= 105
-109 <= matrix[i][j] <= 109
"""