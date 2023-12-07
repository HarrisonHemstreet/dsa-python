import unittest
from math import sqrt, floor

class Sqrt():
    def run(self, num: int) -> int:
        return floor(sqrt(num))

class TestCode(unittest.TestCase):
    def setUp(self):
        self.sqrt = Sqrt()
    
    def test_sqrt(self):
        self.assertEqual(self.sqrt.run(4), 2)
        self.assertEqual(self.sqrt.run(8), 2)

if __name__ == "__main__":
    unittest.main()
"""
69. Sqrt(x)
Easy
7.6K
4.4K
Companies
Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well.

You must not use any built-in exponent function or operator.

For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.
 

Example 1:

Input: x = 4
Output: 2
Explanation: The square root of 4 is 2, so we return 2.
Example 2:

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.
 

Constraints:

0 <= x <= 231 - 1
"""