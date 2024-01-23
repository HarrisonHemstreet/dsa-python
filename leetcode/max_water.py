import unittest

class MaxWater():
    def run(self, height: list[int]) -> int:
        total: int = 0
        i: int = 0
        k: int = len(height) - 1 
        while i < k:
            h: int = min(height[i],height[k])
            width: int = k - i
            area: int = h * width
            if area > total:
                total = area
            if height[i] > height[k]:
                k -= 1
            else:
                i += 1
        return total

class TestCode(unittest.TestCase):
    def setUp(self):
        self.max_water = MaxWater()

    def test_max_water(self):
        self.assertEqual(self.max_water.run([1,8,6,2,5,4,8,3,7]), 49)
        self.assertEqual(self.max_water.run([1,1]), 1)
        self.assertEqual(self.max_water.run([1,2,1]), 2)

if __name__ == "__main__":
    unittest.main()

"""
11. Container With Most Water
Solved
Medium
Topics
Companies
Hint
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

 

Example 1:


Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
Example 2:

Input: height = [1,1]
Output: 1
 

Constraints:

n == height.length
2 <= n <= 105
0 <= height[i] <= 104
"""
