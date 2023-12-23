import unittest

class PathCrossing:
    def run(self, path: str) -> bool:
        pairs: list[tuple[int, int]] = [(0,0)]
        pair: tuple[int, int] = (0,0)

        for char in path:
            x: int = pair[0]
            y: int = pair[1]

            if char == "W":
                pair = (x + 1, y)
            elif char == "N":
                pair = (x, y + 1)
            elif char == "E":
                pair = (x - 1, y)
            elif char == "S":
                pair = (x, y - 1)
            
            if pair not in pairs:
                pairs.append(pair)
            else:
                return True

        return False

class TestCode(unittest.TestCase):
    def setUp(self):
        self.path_crossing = PathCrossing()
    
    def test_path_crossing(self):
        self.assertEqual(self.path_crossing.run("NES"), False)
        self.assertEqual(self.path_crossing.run("NESWW"), True)

if __name__ == "__main__":
    unittest.main()

"""
1496. Path Crossing
Easy
1.3K
39
Companies
Given a string path, where path[i] = 'N', 'S', 'E' or 'W', each representing moving one unit north, south, east, or west, respectively. You start at the origin (0, 0) on a 2D plane and walk on the path specified by path.

Return true if the path crosses itself at any point, that is, if at any time you are on a location you have previously visited. Return false otherwise.

 

Example 1:


Input: path = "NES"
Output: false 
Explanation: Notice that the path doesn't cross any point more than once.
Example 2:


Input: path = "NESWW"
Output: true
Explanation: Notice that the path visits the origin twice.
 

Constraints:

1 <= path.length <= 104
path[i] is either 'N', 'S', 'E', or 'W'.
"""