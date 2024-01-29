import unittest

class MaxVowels():
    def run(self, s: str, k: int) -> int:
        vowels = 'aeiou'
        cur_v = max_v = sum([1 for x in s[:k] if x in vowels])
        for i in range(0, len(s) - k):
            cur_v += (s[i + k] in vowels) - (s[i] in vowels)
            if cur_v > max_v:
                max_v = cur_v
        return max_v

    def count_vowels(self, char_list):
        vowels = set('aeiouAEIOU')  # Set of vowels (including uppercase)
        return sum(char in vowels for char in char_list)

class TestCode(unittest.TestCase):
    def setUp(self):
        self.max_vowels = MaxVowels()

    def test_max_vowels(self):
        self.assertEqual(self.max_vowels.run("abciiidef", 3), 3)
        self.assertEqual(self.max_vowels.run("aeiou", 2), 2)
        self.assertEqual(self.max_vowels.run("leetcode", 3), 2)

if __name__ == "__main__":
    unittest.main()

"""
1456. Maximum Number of Vowels in a Substring of Given Length
Medium
Topics
Companies
Hint
Given a string s and an integer k, return the maximum number of vowel letters in any substring of s with length k.

Vowel letters in English are 'a', 'e', 'i', 'o', and 'u'.

 

Example 1:

Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.
Example 2:

Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.
Example 3:

Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet" and "ode" contain 2 vowels.
 

Constraints:

1 <= s.length <= 105
s consists of lowercase English letters.
1 <= k <= s.length
"""
