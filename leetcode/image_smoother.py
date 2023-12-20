import unittest
from math import floor

class ImageSmoother():
    def run(self, img: list[list[int]]) -> list[list[int]]:
        # Get the dimensions of the image matrix
        rows, cols = len(img), len(img[0])
        
        # Define a helper function to calculate the average value for a pixel
        def average_value(r, c):
            total, count = 0, 0

            # Define the boundaries for the neighboring pixels
            top = max(0, r - 1)
            bottom = min(rows, r + 2)
            left = max(0, c - 1)
            right = min(cols, c + 2)

            # Iterate over the neighboring pixels and calculate the sum and count
            for row in range(top, bottom):
                for col in range(left, right):
                    total += img[row][col]
                    count += 1

            # Calculate and return the average value for the pixel
            return total // count

        # Apply the average function to each pixel in the image matrix
        return [[average_value(r, c) for c in range(cols)] for r in range(rows)]

class TestCode(unittest.TestCase):
    def setUp(self):
        self.image_smoother = ImageSmoother()
    
    def test_image_smoother(self):
        self.assertEqual(self.image_smoother.run([[1,1,1],[1,0,1],[1,1,1]]), [[0,0,0],[0,0,0],[0,0,0]])
        self.assertEqual(self.image_smoother.run([[100,200,100],[200,50,200],[100,200,100]]), [[137,141,137],[141,138,141],[137,141,137]])

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