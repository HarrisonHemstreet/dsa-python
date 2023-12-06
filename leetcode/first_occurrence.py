import unittest

class FirstOccurrence():
    def run(self, haystack: str, needle: str) -> int:
        # 1 loop through
        # 2 find the index of of the first letter of the needle
        # 3 check every place the first letter shows up
        found: int

        try:
            found = haystack.index(needle)
        except:
            found = -1
        
        return found

class TestCode(unittest.TestCase):
    def setUp(self):
        self.first_occurrence = FirstOccurrence()
    
    def test_first_occurrence(self):
        self.assertEqual(self.first_occurrence.run("bear", "ear"), 1)
        self.assertEqual(self.first_occurrence.run("ted", "ed"), 1)
        self.assertEqual(self.first_occurrence.run("bear eat bois", "aoeu"), -1)    
if __name__ == "__main__":
    unittest.main()
"""
28. Find the Index of the First Occurrence in a String

Companies
Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

 

Example 1:

Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.
Example 2:

Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
 

Constraints:

1 <= haystack.length, needle.length <= 104
haystack and needle consist of only lowercase English characters.
"""