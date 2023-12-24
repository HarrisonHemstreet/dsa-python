import unittest

class MinChanges:
    def run(self, s: str) -> int:
        len_s: int = len(s)
        list_s: list[str] = list(s)

        first_op: str = ''.join(["0" if i % 2 == 0 else "1" for i in range(len_s)])
        second_op: str = ''.join(["1" if i % 2 == 0 else "0" for i in range(len_s)])
        first_count: int = 0
        second_count: int = 0

        for i, char in enumerate(list_s):
            if char is not first_op[i]:
                first_count += 1
            if char is not second_op[i]:
                second_count += 1

        return min(first_count, second_count)

class TestMinChanges(unittest.TestCase):
    def setUp(self):
        self.min_changes = MinChanges()
    
    def test_min_changes(self):
        self.assertEqual(self.min_changes.run("0100"),1)
        self.assertEqual(self.min_changes.run("10"),0)
        self.assertEqual(self.min_changes.run("1111"),2)

if __name__ == "__main__":
    unittest.main()

"""
1758. Minimum Changes To Make Alternating Binary String
Easy
1.3K
39
Companies
You are given a string s consisting only of the characters '0' and '1'. In one operation, you can change any '0' to '1' or vice versa.

The string is called alternating if no two adjacent characters are equal. For example, the string "010" is alternating, while the string "0100" is not.

Return the minimum number of operations needed to make s alternating.

 

Example 1:

Input: s = "0100"
Output: 1
Explanation: If you change the last character to '1', s will be "0101", which is alternating.
Example 2:

Input: s = "10"
Output: 0
Explanation: s is already alternating.
Example 3:

Input: s = "1111"
Output: 2
Explanation: You need two operations to reach "0101" or "1010".
 

Constraints:

1 <= s.length <= 104
s[i] is either '0' or '1'.
"""