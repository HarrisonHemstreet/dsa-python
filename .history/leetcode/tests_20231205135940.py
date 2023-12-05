from two_sum import TwoSum

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.two_sum = TwoSum()

    def test_two_sum(self):
        self.assertEqual(self.two_sum.two_sum([2,7,11,15], 9), [0,1])
        self.assertEqual(self.two_sum.two_sum([3,2,4], 6), [1,2])
        self.assertEqual(self.solution.two_sum([3,3], 6), [0,1])

if __name__ == '__main__':
    unittest.main()
