import unittest
from two_sum import TwoSum
from palindrome_number import PalindromeNumber
from roman_to_int import RomanToInt
from longest_common_prefix import LongestCommonPrefix
from valid_parens import ValidParens
from remove_duplicates import RemoveDups
from remove_element import RemoveElement
from first_occurrence import FirstOccurrence
from search_insert import SearchInsert
from length_of_last_word import LengthLastWord
from plus_one import PlusOne
from add_binary import AddBinary

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.two_sum = TwoSum()
        self.palindrome_number = PalindromeNumber()
        self.roman_to_int = RomanToInt()
        self.longest_common_prefix = LongestCommonPrefix()
        self.valid_parens = ValidParens()
        self.remove_dups = RemoveDups()
        self.remove_element = RemoveElement()
        self.first_occurrence = FirstOccurrence()
        self.search_insert = SearchInsert()
        self.length_last_word = LengthLastWord()
        self.plus_one = PlusOne()
        self.add_binary = AddBinary()

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
    
    def test_remove_element(self):
        self.assertEqual(self.remove_element.run([3,2,2,3], 3), 2)
        self.assertEqual(self.remove_element.run([0,1,2,2,3,0,4,2], 2), 5)
    
    def test_first_occurrence(self):
        self.assertEqual(self.first_occurrence.run("bear", "ear"), 1)
        self.assertEqual(self.first_occurrence.run("ted", "ed"), 1)
        self.assertEqual(self.first_occurrence.run("bear eat bois", "aoeu"), -1)
    
    def test_search_insert(self):
        self.assertEqual(self.search_insert.run([1,3,5,6], 5), 2)
        self.assertEqual(self.search_insert.run([1,3,5,6], 2), 1)
        self.assertEqual(self.search_insert.run([1,3,5,6], 7), 4)
        self.assertEqual(self.search_insert.run([1], 0), 0)
        self.assertEqual(self.search_insert.run([1], 1), 0)
        self.assertEqual(self.search_insert.run([1,3], 2), 1)
    
    def test_length_last_word(self):
        self.assertEqual(self.length_last_word.run("Hello World"), 5)
        self.assertEqual(self.length_last_word.run("   fly me   to   the moon  "), 4)
        self.assertEqual(self.length_last_word.run("luffy is still joyboy"), 6)
    
    def test_plus_one(self):
        self.assertEqual(self.plus_one.run([1,2,3]), [1,2,4])
        self.assertEqual(self.plus_one.run([4,3,2,1]), [4,3,2,2])
        self.assertEqual(self.plus_one.run([9]), [1,0])
        self.assertEqual(self.plus_one.run([1,0]), [1,1])
    
    def test_add_binary(self):
        self.assertEqual(self.add_binary.run("11", "1"), "100")
        self.assertEqual(self.add_binary.run("1010", "1011"), "10101")

if __name__ == '__main__':
    unittest.main()
