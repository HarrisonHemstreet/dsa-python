import unittest
from math import floor

class ImageSmoother():
    def run(self, img: list[list[int]]) -> list[list[int]]:
        """
        1. get every group of nums
        2. output the average of every group of nums
        
        square: 3x3:
        # C = img[i][k]
        """
        rows_len: int = len(img)
        cols_len: int = len(img[0])

        i: int = rows_len
        k: int = cols_len

        while i >= 0:
            i -= 1

            while k >= 0:
                k -= 1

                TL: int | None
                TM: int | None
                TR: int | None
                ML: int | None
                MM: int | None
                MR: int | None
                BL: int | None
                BM: int | None
                BR: int | None

                try:
                    TL = img[i - 1][k - 1]
                except IndexError:
                    TL = None
                try:
                    TM = img[i - 1][k]
                except IndexError:
                    TM = None
                try:
                    TR = img[i - 1][k + 1]
                except IndexError:
                    TR = None
                try:
                    ML = img[i][k - 1]
                except IndexError:
                    ML = None
                try:
                    MM = img[i][k]
                except IndexError:
                    MM = None
                try:
                    MR = img[i][k + 1]
                except IndexError:
                    MR = None
                try:
                    BL = img[i + 1][k - 1]
                except IndexError:
                    BL = None
                try:
                    BM = img[i + 1][k]
                except IndexError:
                    BM = None
                try:
                    BR = img[i + 1][k + 1]
                except IndexError:
                    BR = None

                grid = [TL, TM, TR, ML, MM, MR, BL, BM, BR]
                to_smooth = [num for num in grid if num is not None]
                smooth_len: int = len(to_smooth)
                img[i][k] = floor(sum(to_smooth) / smooth_len)


class TestCode(unittest.TestCase):
    def setUp(self):
        self.image_smoother = ImageSmoother()
    
    def test_image_smoother(self):
        self.assertEqual(self.image_smoother.run())

if __name__ == "__main__":
    unittest.main()


"""
[
    [ 1, 2,  3,  4,  5],
    [ 6, 7,  8,  9,  10],
    [11, 12, 13, 14, 15]
]
661. Image Smoother
Easy
1K
2.8K
Companies
An image smoother is a filter of the size 3 x 3 that can be applied to each cell of an image by rounding down the average of the cell and the eight surrounding cells (i.e., the average of the nine cells in the blue smoother). If one or more of the surrounding cells of a cell is not present, we do not consider it in the average (i.e., the average of the four cells in the red smoother).


Given an m x n integer matrix img representing the grayscale of an image, return the image after applying the smoother on each cell of it.

 

Example 1:


Input: img = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[0,0,0],[0,0,0],[0,0,0]]
Explanation:
For the points (0,0), (0,2), (2,0), (2,2): floor(3/4) = floor(0.75) = 0
For the points (0,1), (1,0), (1,2), (2,1): floor(5/6) = floor(0.83333333) = 0
For the point (1,1): floor(8/9) = floor(0.88888889) = 0
Example 2:


Input: img = [[100,200,100],[200,50,200],[100,200,100]]
Output: [[137,141,137],[141,138,141],[137,141,137]]
Explanation:
For the points (0,0), (0,2), (2,0), (2,2): floor((100+200+200+50)/4) = floor(137.5) = 137
For the points (0,1), (1,0), (1,2), (2,1): floor((200+200+50+200+100+100)/6) = floor(141.666667) = 141
For the point (1,1): floor((50+200+200+200+200+100+100+100+100)/9) = floor(138.888889) = 138
 

Constraints:

m == img.length
n == img[i].length
1 <= m, n <= 200
0 <= img[i][j] <= 255
"""