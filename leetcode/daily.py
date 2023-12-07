# 06 December 2023

# Time:  O(n)
# Space: O(h)

class Node(object):
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Solution(object):
    def construct(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: Node
        """
        def dfs(grid, x, y, l):
            if l == 1:
                return Node(grid[x][y] == 1, True, None, None, None, None)
            half = l // 2
            topLeftNode = dfs(grid, x, y, half)
            topRightNode = dfs(grid, x, y+half, half)
            bottomLeftNode = dfs(grid, x+half, y, half)
            bottomRightNode = dfs(grid, x+half, y+half, half)
            if topLeftNode.isLeaf and topRightNode.isLeaf and \
               bottomLeftNode.isLeaf and bottomRightNode.isLeaf and \
               topLeftNode.val == topRightNode.val == bottomLeftNode.val == bottomRightNode.val:
                return Node(topLeftNode.val, True, None, None, None, None)
            return Node(True, False, topLeftNode, topRightNode, bottomLeftNode, bottomRightNode)
        
        if not grid:
            return None
        return dfs(grid, 0, 0, len(grid))

# 07 December 2023

# Time:  O(n)
# Space: O(n)

import collections


class Solution(object):
    def getDistances(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        lookup = collections.defaultdict(list)
        for i, x in enumerate(arr):
            lookup[x].append(i)
        result = [0]*len(arr)
        for idxs in lookup.itervalues():
            prefix = [0]
            for i in idxs:
                prefix.append(prefix[-1]+i)
            for i, idx in enumerate(idxs):
                result[idx] = (idx*(i+1)-prefix[i+1]) + ((prefix[len(idxs)]-prefix[i])-idx*(len(idxs)-i))
        return result

