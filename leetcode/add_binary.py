import unittest

class AddBinary():
    def run(self, a: str, b: str) -> str:
        a: int = int(a, 2)
        b: int = int(b, 2)
        b_sum: int = a + b
        return format(b_sum, 'b')

class TestCode(unittest.TestCase):
    def setUp(self):
        self.add_binary = AddBinary()
    
    def test_add_binary(self):
        self.assertEqual(self.add_binary.run("11", "1"), "100")
        self.assertEqual(self.add_binary.run("1010", "1011"), "10101")

if __name__ == '__main__':
    unittest.main()

"""
67. Add Binary
Easy
9K
902
Companies
Given two binary strings a and b, return their sum as a binary string.

 

Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"
 

Constraints:

1 <= a.length, b.length <= 104
a and b consist only of '0' or '1' characters.
Each string does not contain leading zeros except for the zero itself.
"""