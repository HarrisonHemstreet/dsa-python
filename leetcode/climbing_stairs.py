import unittest

class ClimbingStairs():
    def run(self, n: int) -> int:
        memo = {}
        def dp(total):
            if total == n:
                return 1
            
            if total > n:
                return 0

            if total in memo:
                return memo[total]

            memo[total] = dp(total + 1) + dp(total + 2)
            return memo[total]

        return dp(0)

class TestCode(unittest.TestCase):
    def setUp(self):
        self.climbing_stairs = ClimbingStairs()

    def test_climbing_stairs(self):
        self.assertEqual(self.climbing_stairs.run(2), 2)
        self.assertEqual(self.climbing_stairs.run(3), 3)

if __name__ == "__main__":
    unittest.main()

"""
70. Climbing Stairs
Easy
20.5K
710
Companies
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
 

Constraints:

1 <= n <= 45
"""