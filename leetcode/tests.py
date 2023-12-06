import unittest
from two_sum import TwoSum
from palindrome_number import PalindromeNumber
from roman_to_int import RomanToInt
from longest_common_prefix import LongestCommonPrefix
from valid_parens import ValidParens
from remove_duplicates import RemoveDups

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.two_sum = TwoSum()
        self.palindrome_number = PalindromeNumber()
        self.roman_to_int = RomanToInt()
        self.longest_common_prefix = LongestCommonPrefix()
        self.valid_parens = ValidParens()
        self.remove_dups = RemoveDups()

    def test_two_sum(self):
        self.assertEqual(self.two_sum.run([2,7,11,15], 9), [0,1])
        self.assertEqual(self.two_sum.run([3,2,4], 6), [1,2])
        self.assertEqual(self.two_sum.run([3,3], 6), [0,1])
    
    def test_palindrome_number(self):
        self.assertEqual(self.palindrome_number.run(121), True)
        self.assertEqual(self.palindrome_number.run(-121), False)
        self.assertEqual(self.palindrome_number.run(123), False)
    
    def test_roman_to_int(self):
        self.assertEqual(self.roman_to_int.run("III"), 3)
        self.assertEqual(self.roman_to_int.run("LVIII"), 58)
        self.assertEqual(self.roman_to_int.run("MCMXCIV"), 1994)
    
    def test_longest_common_prefix(self):
        self.assertEqual(self.longest_common_prefix.run(["flower","flow","flight"]), "fl")
        self.assertEqual(self.longest_common_prefix.run(["dog","racecar","car"]), "")
        self.assertEqual(self.longest_common_prefix.run(["tom","tooth","toy"]), "to")
        self.assertEqual(self.longest_common_prefix.run(["cir","car"]), "c")

    def test_valid_parens(self):
        self.assertEqual(self.valid_parens.run("()"), True)
        self.assertEqual(self.valid_parens.run("()[]{}"), True)
        self.assertEqual(self.valid_parens.run("(]"), False)
        self.assertEqual(self.valid_parens.run("([)]"), False)
        self.assertEqual(self.valid_parens.run("({[]})"), True)
    
    def test_remove_dups(self):
        self.assertEqual(self.remove_dups.run([1,1,2,3,4,4]), 4)
        self.assertEqual(self.remove_dups.run([1,2,2,3,4,4]), 4)
        self.assertEqual(self.remove_dups.run([1,2]), 2)
        self.assertEqual(self.remove_dups.run([0,0,1,1,1,2,2,3,3,4]), 5)

if __name__ == '__main__':
    unittest.main()
