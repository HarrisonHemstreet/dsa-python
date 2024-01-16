import unittest

class ReverseVowels():
    def run(self, s: str) -> str:
        list_s: list[str] = list(s)
        len_s: int = len(s)
        i: int = 0
        k: int = len_s - 1

        vowels: list[str] = ["a","e","i","o","u"]

        while i < k:
            left: str = list_s[i].lower()
            right: str = list_s[k].lower()
            if left in vowels and right in vowels:
                list_s[i], list_s[k] = list_s[k], list_s[i]
                k -= 1
                i += 1
            elif left in vowels:
                k -= 1
            elif right in vowels:
                i += 1
            else:
                k -= 1
                i += 1
        return "".join(list_s)

class TestCode(unittest.TestCase):
    def setUp(self):
        self.reverse_vowels = ReverseVowels()

    def test_reverse_vowels(self):
        self.assertEqual(self.reverse_vowels.run("hello"), "holle")
        self.assertEqual(self.reverse_vowels.run("leetcode"), "leotcede")
        self.assertEqual(self.reverse_vowels.run("aA"), "Aa")

if __name__ == "__main__":
    unittest.main()

"""
345. Reverse Vowels of a String
Easy
Topics
Companies
Given a string s, reverse only all the vowels in the string and return it.

The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in both lower and upper cases, more than once.

 

Example 1:

Input: s = "hello"
Output: "holle"
Example 2:

Input: s = "leetcode"
Output: "leotcede"
 

Constraints:

1 <= s.length <= 3 * 105
s consist of printable ASCII characters.
"""
