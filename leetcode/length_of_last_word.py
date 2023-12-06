import unittest

class LengthLastWord:
    def run(self, s: str) -> int:
        s = s.strip()
        i: int = len(s) - 1
        count: int = 0
        while i >= 0:
            if s[i] == " ":
                return count
            count += 1
            i -= 1
        return count

class TestCode(unittest.TestCase):
    def setUp(self):
        self.length_last_word = LengthLastWord()
    
    def test_length_last_word(self):
        self.assertEqual(self.length_last_word.run("Hello World"), 5)
        self.assertEqual(self.length_last_word.run("   fly me   to   the moon  "), 4)
        self.assertEqual(self.length_last_word.run("luffy is still joyboy"), 6)

if __name__ == "__main__":
    unittest.main()

"""
58. Length of Last Word
Easy
4.3K
224
Companies
Given a string s consisting of words and spaces, return the length of the last word in the string.

A word is a maximal 
substring
 consisting of non-space characters only.

 

Example 1:

Input: s = "Hello World"
Output: 5
Explanation: The last word is "World" with length 5.
Example 2:

Input: s = "   fly me   to   the moon  "
Output: 4
Explanation: The last word is "moon" with length 4.
Example 3:

Input: s = "luffy is still joyboy"
Output: 6
Explanation: The last word is "joyboy" with length 6.
 

Constraints:

1 <= s.length <= 104
s consists of only English letters and spaces ' '.
There will be at least one word in s.
"""