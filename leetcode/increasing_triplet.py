import unittest

class IncreasingTriplet():
    def run(self, nums: list[int]) -> bool:
        """
        i: int = 0
        k: int = nums[0]
        while i < len(nums):
            curr: int = nums[i]
            if 
        """
        first: float | int = float('inf')
        second: float | int = first
        for num in nums:
            if num < first:
                first = num
            elif num > first and num < second:
                second = num
            elif num > second:
                return True

        return False

# 

class TestCode(unittest.TestCase):
    def setUp(self):
        self.increasing_triplet = IncreasingTriplet()

    def test_increasing_triplet(self):
        self.assertEqual(self.increasing_triplet.run([1,2,3,4,5]), True)
        self.assertEqual(self.increasing_triplet.run([5,4,3,2,1]), False)
        self.assertEqual(self.increasing_triplet.run([2,1,5,0,4,6]), True)
        self.assertEqual(self.increasing_triplet.run([1,2,1,3]), True)
        self.assertEqual(self.increasing_triplet.run([20,100,10,12,5,13]), True)

if __name__ == "__main__":
    unittest.main()

"""
334. Increasing Triplet Subsequence
Medium
Topics
Companies
Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.

 

Example 1:

Input: nums = [1,2,3,4,5]
Output: true
Explanation: Any triplet where i < j < k is valid.
Example 2:

Input: nums = [5,4,3,2,1]
Output: false
Explanation: No triplet exists.
Example 3:

Input: nums = [2,1,5,0,4,6]
Output: true
Explanation: The triplet (3, 4, 5) is valid because nums[3] == 0 < nums[4] == 4 < nums[5] == 6.
 

Constraints:

1 <= nums.length <= 5 * 105
-231 <= nums[i] <= 231 - 1
"""
