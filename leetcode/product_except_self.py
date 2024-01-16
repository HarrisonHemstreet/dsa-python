import unittest
from math import prod

class ProductExceptSelf():
    def run(self, nums: list[int]) -> list[int]:
        zero_count: int = nums.count(0)
        if zero_count > 1:
            return [0] * len(nums)
        total: int = prod(nums)
        res: list[int] = []
        for i, num in enumerate(nums):
            if num == 0:
                sub_prod: int = 1
                for num in nums:
                    if num != 0:
                        sub_prod *= num
                res.append(sub_prod)
            else:
                res.append(total // num)
        return res

class TestCode(unittest.TestCase):
    def setUp(self):
        self.product_except_self = ProductExceptSelf()

    def test_product_except_self(self):
        self.assertEqual(self.product_except_self.run([1,2,3,4]), [24,12,8,6])
        self.assertEqual(self.product_except_self.run([-1,1,0,-3,3]), [0,0,9,0,0])

if __name__ == "__main__":
    unittest.main()
"""
238. Product of Array Except Self
Solved
Medium
Topics
Companies
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

 

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
 

Constraints:

2 <= nums.length <= 105
-30 <= nums[i] <= 30
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
 

Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)
"""
