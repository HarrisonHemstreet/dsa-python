import unittest

class MoveZeroes():
    def run(self, nums: list[int]) -> list[int]:
        l = 0
        r = 0

        while r < len(nums):
            if nums[r] != 0:
                temp = nums[l]
                nums[l] = nums[r]
                nums[r] = temp
                l += 1
            r += 1
        return nums

class TestCode(unittest.TestCase):
    def setUp(self):
        self.move_zeroes = MoveZeroes()

    def test_move_zeroes(self):
        self.assertEqual(self.move_zeroes.run([0,1,0,3,12]), [1,3,12,0,0])
        self.assertEqual(self.move_zeroes.run([0]), [0])
        self.assertEqual(self.move_zeroes.run([0,0,1]), [1,0,0])

if __name__ == "__main__":
    unittest.main()
"""
283. Move Zeroes
Solved
Easy
Topics
Companies
Hint
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.

 

Example 1:

Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
Example 2:

Input: nums = [0]
Output: [0]
 

Constraints:

1 <= nums.length <= 104
-231 <= nums[i] <= 231 - 1
 

Follow up: Could you minimize the total number of operations done?
"""
