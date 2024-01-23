import unittest

class IsSubsequence():
    def run(self, s: str, t: str) -> bool:
        t_len: int = len(t)
        i: int = 0

        k: int = 0
        sub_len: int = len(s)

        count: int = 0

        if sub_len == 0:
            return True

        while i < t_len:
            curr: str = t[i]
            curr_sub: str = s[k]
            if curr == curr_sub:
                count += 1
                k += 1
            if count >= sub_len:
                return True
            i += 1
        return False

class TestCode(unittest.TestCase):
    def setUp(self):
        self.is_subsequence = IsSubsequence()

    def test_is_subsequence(self):
        self.assertEqual(self.is_subsequence.run("abc","ahbgdc"),True)
        self.assertEqual(self.is_subsequence.run("axc","ahbgdc"),False)
        self.assertEqual(self.is_subsequence.run("","ahbgdc"),True)

if __name__ == "__main__":
    unittest.main()

"""
392. Is Subsequence
Easy
Topics
Companies
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

 

Example 1:

Input: s = "abc", t = "ahbgdc"
Output: true
Example 2:

Input: s = "axc", t = "ahbgdc"
Output: false
 

Constraints:

0 <= s.length <= 100
0 <= t.length <= 104
s and t consist only of lowercase English letters.
 

Follow up: Suppose there are lots of incoming s, say s1, s2, ..., sk where k >= 109, and you want to check one by one to see if t has its subsequence. In this scenario, how would you change your code?
"""
