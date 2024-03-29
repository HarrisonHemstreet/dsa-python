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
from largest_odd_num_in_str import LargestOddNumInStr
from sqrt import Sqrt
from climbing_stairs import ClimbingStairs
from delete_dups import DeleteDups
from nodes import ListNode, tree1, tree2, tree3, tree7, tree8, tree9, tree10
from element_25_percent_of_array import TwentyFivePercentOfArray
from merge_sorted_array import MergeSortedArray
from max_product_two_elms_in_list import MaxProduct
from same_tree import SameTree
from symmetric_tree import SymmetricTree
from max_prod_two_pairs import MaxProdTwoPairs
from image_smoother import ImageSmoother
from buy_two_chocolates import BuyTwoChocolates
from path_crossing import PathCrossing
from min_changes import MinChanges
from gcd_of_strings import GCDOfStrings
from merge_strings_alt import MergeAlternately
from kids_with_candies import KidsWithCandies
from can_place_flowers import CanPlaceFlowers
from reverse_vowels import ReverseVowels
from reverse_words import ReverseWords
from product_except_self import ProductExceptSelf
from increasing_triplet import IncreasingTriplet
from compress_string import CompressString
from move_zeroes import MoveZeroes
from is_subsequence import IsSubsequence
from max_water import MaxWater
from max_ops import MaxOps
from find_max_average import FindMaxAverage
from max_vowels import MaxVowels
from max_odd_binary_number import MaxOddBinaryNumber

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
        self.largest_odd_num_in_str = LargestOddNumInStr()
        self.sqrt = Sqrt()
        self.climbing_stairs = ClimbingStairs()
        self.delete_dups = DeleteDups()
        self.twenty_five_percent = TwentyFivePercentOfArray()
        self.merge_sorted_array = MergeSortedArray()
        self.max_prod = MaxProduct()
        self.same_tree = SameTree()
        self.symmetric_tree = SymmetricTree()
        self.max_prod_two_pairs = MaxProdTwoPairs()
        self.image_smoother = ImageSmoother()
        self.buy_two_chocolates = BuyTwoChocolates()
        self.path_crossing = PathCrossing()
        self.min_changes = MinChanges()
        self.gcd_of_strings = GCDOfStrings()
        self.merge_alternately = MergeAlternately()
        self.kids_with_candies = KidsWithCandies()
        self.can_place_flowers = CanPlaceFlowers()
        self.reverse_vowels = ReverseVowels()
        self.reverse_words = ReverseWords()
        self.product_except_self = ProductExceptSelf()
        self.increasing_triplet = IncreasingTriplet()
        self.compress_string = CompressString()
        self.move_zeroes = MoveZeroes()
        self.is_subsequence = IsSubsequence()
        self.max_water = MaxWater()
        self.max_ops = MaxOps()
        self.find_max_average = FindMaxAverage()
        self.max_vowels = MaxVowels()
        self.max_odd = MaxOddBinaryNumber()

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

    def test_largest_odd_num_in_str(self):
        self.assertEqual(self.largest_odd_num_in_str.run("52"), "5")
        self.assertEqual(self.largest_odd_num_in_str.run("4206"), "")
        self.assertEqual(self.largest_odd_num_in_str.run("35427"), "35427")
        self.assertEqual(self.largest_odd_num_in_str.run("10133890"), "1013389")
    
    def test_sqrt(self):
        self.assertEqual(self.sqrt.run(4), 2)
        self.assertEqual(self.sqrt.run(8), 2)
    
    def test_climbing_stairs(self):
        self.assertEqual(self.climbing_stairs.run(2), 2)
        self.assertEqual(self.climbing_stairs.run(3), 3)
    
    def test_delete_dups(self):
        L1: ListNode = ListNode(1)
        L1_tail: ListNode = L1
        L1_tail.next = ListNode(1)
        L1_tail = L1_tail.next
        L1_tail.next = ListNode(2)
        L1_tail = L1_tail.next

        L1_res: ListNode = ListNode(1)
        L1_tail_res: ListNode = L1_res
        L1_tail_res.next = ListNode(2)
        L1_tail_res = L1_tail_res.next

        L2: ListNode = ListNode(1)
        L2_tail: ListNode = L2
        L2_tail.next = ListNode(1)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(1)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(2)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(3)
        L2_tail = L2_tail.next

        L2: ListNode = ListNode(1)
        L2_tail: ListNode = L2
        L2_tail.next = ListNode(2)
        L2_tail = L2_tail.next
        L2_tail.next = ListNode(3)
        L2_tail = L2_tail.next

        L3: ListNode = ListNode(1)
        L3_tail: ListNode = L3
        L3_tail.next = ListNode(1)
        L3_tail = L3_tail.next
        L3_tail.next = ListNode(1)
        L3_tail = L3_tail.next

        self.assertEqual(self.delete_dups.LL_to_list(L1), [1,2])
        self.assertEqual(self.delete_dups.LL_to_list(L2), [1,2,3])
        self.assertEqual(self.delete_dups.LL_to_list(L3), [1])
    
    def test_twenty_five_percent(self):
        self.assertEqual(self.twenty_five_percent.run([1,2,2,6,6,6,6,7,10]), 6)
        self.assertEqual(self.twenty_five_percent.run([1,1]), 1)
    
    def test_merge_sorted_array(self):
        self.assertEqual(self.merge_sorted_array.run([1,2,3,0,0,0], 3, [2,5,6], 3), [1,2,2,3,5,6])
        self.assertEqual(self.merge_sorted_array.run([1], 1, [], 0), [1])
        self.assertEqual(self.merge_sorted_array.run([0], 0, [1], 1), [1])
    
    def test_max_prod(self):
        self.assertEqual(self.max_prod.run([3,4,5,2]), 12)
        self.assertEqual(self.max_prod.run([1,5,4,5]), 16)
        self.assertEqual(self.max_prod.run([3,7]), 12)
    
    def test_same_tree(self):
        self.assertEqual(self.same_tree.run(tree1, tree1), True)
        self.assertEqual(self.same_tree.run(tree1, tree2), False)
        self.assertEqual(self.same_tree.run(tree2, tree3), False)
        self.assertEqual(self.same_tree.run(tree7, tree8), False)
    
    def test_symmetric_tree(self):
        self.assertEqual(self.symmetric_tree.run(tree9), True)
        self.assertEqual(self.symmetric_tree.run(tree10), False)
    
    def test_max_prod_two_pairs(self):
        self.assertEqual(self.max_prod_two_pairs.run([5,6,2,7,4]), 34)
        self.assertEqual(self.max_prod_two_pairs.run([4,2,5,9,7,4,8]), 64)
    
    def test_image_smoother(self):
        self.assertEqual(self.image_smoother.run([[1,1,1],[1,0,1],[1,1,1]]), [[0,0,0],[0,0,0],[0,0,0]])
        self.assertEqual(self.image_smoother.run([[100,200,100],[200,50,200],[100,200,100]]), [[137,141,137],[141,138,141],[137,141,137]])
    
    def test_buy_two_chocolates(self):
        self.assertEqual(self.buy_two_chocolates.run([1,2,2], 3), 0)
        self.assertEqual(self.buy_two_chocolates.run([3,2,3], 3), 3)
    
    def test_path_crossing(self):
        self.assertEqual(self.path_crossing.run("NES"), False)
        self.assertEqual(self.path_crossing.run("NESWW"), True)
    
    def test_min_changes(self):
        self.assertEqual(self.min_changes.run("0100"),1)
        self.assertEqual(self.min_changes.run("10"),0)
        self.assertEqual(self.min_changes.run("1111"),2)

    def test_gcd_of_strings(self):
        self.assertEqual(self.gcd_of_strings.run("ABCABC", "ABC"), "ABC")
        self.assertEqual(self.gcd_of_strings.run("ABABAB", "ABAB"), "AB")
        self.assertEqual(self.gcd_of_strings.run("LEET", "CODE"), "")

    def test_merge_alternately(self):
        self.assertEqual(self.merge_alternately.run("abc", "pqr"), "apbqcr")
        self.assertEqual(self.merge_alternately.run("ab", "pqrs"), "apbqrs")
        self.assertEqual(self.merge_alternately.run("abcd", "pq"), "apbqcd")

    def test_kids_with_candies(self):
        self.assertEqual(self.kids_with_candies.run([2,3,5,1,3], 3), [True,True,True,False,True])
        self.assertEqual(self.kids_with_candies.run([4,2,1,1,2], 1), [True,False,False,False,False])
        self.assertEqual(self.kids_with_candies.run([12,1,12], 10), [True,False,True])

    def test_can_place_flowers(self):
        self.assertEqual(self.can_place_flowers.run([1,0,0,0,1], 1), True)
        self.assertEqual(self.can_place_flowers.run([1,0,0,0,1], 2), False)
        self.assertEqual(self.can_place_flowers.run([1,0,0,0,0,0,1], 2), True)
        self.assertEqual(self.can_place_flowers.run([1,0,0,0,0,1], 2), False)
        self.assertEqual(self.can_place_flowers.run([1,0,1,0,1,0,1], 1), False)
        self.assertEqual(self.can_place_flowers.run([0,0,1,0,1], 1), True)
        self.assertEqual(self.can_place_flowers.run([0], 1), True)
        self.assertEqual(self.can_place_flowers.run([0,0,1,0,0], 1), True)

    def test_reverse_vowels(self):
        self.assertEqual(self.reverse_vowels.run("hello"), "holle")
        self.assertEqual(self.reverse_vowels.run("leetcode"), "leotcede")
        self.assertEqual(self.reverse_vowels.run("aA"), "Aa")

    def test_reverse_words(self):
        self.assertEqual(self.reverse_words.run("the sky is blue"), "blue is sky the")
        self.assertEqual(self.reverse_words.run("  hello world  "), "world hello")
        self.assertEqual(self.reverse_words.run("a good   example"), "example good a")

    def test_product_except_self(self):
        self.assertEqual(self.product_except_self.run([1,2,3,4]), [24,12,8,6])
        self.assertEqual(self.product_except_self.run([-1,1,0,-3,3]), [0,0,9,0,0])

    def test_increasing_triplet(self):
        self.assertEqual(self.increasing_triplet.run([1,2,3,4,5]), True)
        self.assertEqual(self.increasing_triplet.run([5,4,3,2,1]), False)
        self.assertEqual(self.increasing_triplet.run([2,1,5,0,4,6]), True)
        self.assertEqual(self.increasing_triplet.run([1,2,1,3]), True)
        self.assertEqual(self.increasing_triplet.run([20,100,10,12,5,13]), True)

    def test_compress_string(self):
        self.assertEqual(self.compress_string.run(["a","a","b","b","c","c","c"]), 6)
        self.assertEqual(self.compress_string.run(["a"]), 1)
        self.assertEqual(self.compress_string.run(["a","b","b","b","b","b","b","b","b","b","b","b","b"]), 4)

    def test_move_zeroes(self):
        self.assertEqual(self.move_zeroes.run([0,1,0,3,12]), [1,3,12,0,0])
        self.assertEqual(self.move_zeroes.run([0]), [0])
        self.assertEqual(self.move_zeroes.run([0,0,1]), [1,0,0])

    def test_is_subsequence(self):
        self.assertEqual(self.is_subsequence.run("abc","ahbgdc"),True)
        self.assertEqual(self.is_subsequence.run("axc","ahbgdc"),False)
        self.assertEqual(self.is_subsequence.run("","ahbgdc"),True)

    def test_max_water(self):
        self.assertEqual(self.max_water.run([1,8,6,2,5,4,8,3,7]), 49)
        self.assertEqual(self.max_water.run([1,1]), 1)
        self.assertEqual(self.max_water.run([1,2,1]), 2)
    
    def test_max_ops(self):
        self.assertEqual(self.max_ops.run([1,2,3,4], 5), 2)
        self.assertEqual(self.max_ops.run([3,1,3,4,3], 6), 1)

    def test_find_max_average(self):
        self.assertEqual(self.find_max_average.run([1,12,-5,-6,50,3], 4), 12.75000)
        self.assertEqual(self.find_max_average.run([5], 1), 5)
        self.assertEqual(self.find_max_average.run([-1], 1), -1.0)
        self.assertEqual(self.find_max_average.run(
            [8860,-853,6534,4477,-4589,8646,-6155,-5577,-1656,-5779,-2619,-8604,-1358,-8009,4983,7063,3104,-1560,4080,2763,5616,-2375,2848,1394,-7173,-5225,-8244,-809,8025,-4072,-4391,-9579,1407,6700,2421,-6685,5481,-1732,-8892,-6645,3077,3287,-4149,8701,-4393,-9070,-1777,2237,-3253,-506,-4931,-7366,-8132,5406,-6300,-275,-1908,67,3569,1433,-7262,-437,8303,4498,-379,3054,-6285,4203,6908,4433,3077,2288,9733,-8067,3007,9725,9669,1362,-2561,-4225,5442,-9006,-429,160,-9234,-4444,3586,-5711,-9506,-79,-4418,-4348,-5891],
            93
        ), -594.5806451612904)
        self.assertEqual(self.find_max_average.run([5], 1), 5)

    def test_max_vowels(self):
        self.assertEqual(self.max_vowels.run("abciiidef", 3), 3)
        self.assertEqual(self.max_vowels.run("aeiou", 2), 2)
        self.assertEqual(self.max_vowels.run("leetcode", 3), 2)

    def test_max_odd(self):
        self.assertEqual(self.max_odd.run("100"), "001")
        self.assertEqual(self.max_odd.run("010"), "001")
        self.assertEqual(self.max_odd.run("0101"), "1001")

if __name__ == '__main__':
    unittest.main()
