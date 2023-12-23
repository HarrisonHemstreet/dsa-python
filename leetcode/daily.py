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

# 08 December 2023

# Time:  O(n)
# Space: O(1)

import collections
import string

class Solution(object):
    def minDeletions(self, s):
        """
        :type s: str
        :rtype: int
        """
        count = collections.Counter(s)
        result = 0
        lookup = set()
        for c in string.ascii_lowercase:
            for i in reversed(xrange(1, count[c]+1)):
                if i not in lookup:
                    lookup.add(i)
                    break
                result += 1
        return result

# 09 December 2023

# Time:  O(n)
# Space: O(1)

class Solution(object):
    # @param {string[]} words
    # @param {string} word1
    # @param {string} word2
    # @return {integer}
    def shortestWordDistance(self, words, word1, word2):
        dist = float("inf")
        is_same = (word1 == word2)
        i, index1, index2 = 0, None, None
        while i < len(words):
            if words[i] == word1:
                if is_same and index1 is not None:
                    dist = min(dist, abs(index1 - i))
                index1 = i
            elif words[i] == word2:
                index2 = i

            if index1 is not None and index2 is not None:
                dist = min(dist, abs(index1 - index2))
            i += 1

        return dist

# 10 December 2023

# Time:  O(n)
# Space: O(h)

class Solution(object):
    def pruneTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if not root.left and not root.right and root.val == 0:
            return None
        return root

# 11 December 2023

# Time:  O(n * (n/k)!)
# Space: O(n)

import collections


class Solution(object):
    def longestSubsequenceRepeatedK(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        def check(s, k, curr):
            if not curr:
                return True
            i = 0
            for c in s:
                if c != curr[i]:
                    continue
                i += 1
                if i != len(curr):
                    continue
                i = 0
                k -= 1
                if not k:
                    return True
            return False

        def backtracking(s, k, curr, cnts, result):
            if not check(s, k, curr):
                return
            if len(curr) > len(result):
                result[:] = curr
            for c in reversed(string.ascii_lowercase):
                if cnts[c] < k:
                    continue
                cnts[c] -= k
                curr.append(c)
                backtracking(s, k, curr, cnts, result)
                curr.pop()
                cnts[c] += k
                    
        cnts = collections.Counter(s)
        new_s = []
        for c in s:
            if cnts[c] < k:
                continue
            new_s.append(c)
        result =[]
        backtracking(new_s, k, [], cnts, result)
        return "".join(result)

# 12 December 2023

# Time:  O(nlogn)
# Space: O(n)

from collections import Counter
from heapq import heapify, heappop


class Solution(object):
    def isNStraightHand(self, hand, W):
        """
        :type hand: List[int]
        :type W: int
        :rtype: bool
        """
        if len(hand) % W:
            return False

        counts = Counter(hand)
        min_heap = list(hand)
        heapify(min_heap)
        for _ in xrange(len(min_heap)//W):
            while counts[min_heap[0]] == 0:
                heappop(min_heap)
            start = heappop(min_heap)
            for _ in xrange(W):
                counts[start] -= 1
                if counts[start] < 0:
                    return False
                start += 1
        return True

# 13 December 2023

# Time:  O(nlogn)
# Space: O(1)

class Solution(object):
    def eliminateMaximum(self, dist, speed):
        """
        :type dist: List[int]
        :type speed: List[int]
        :rtype: int
        """
        for i in xrange(len(dist)):
            dist[i] = (dist[i]-1)//speed[i]
        dist.sort()
        result = 0
        for i in xrange(len(dist)):
            if result > dist[i]:
                break
            result += 1
        return result

# 14 December 2023

# Time:  O(n)
# Space: O(h)

import random


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        pass


# bfs, quick select
class Solution(object):
    def kthLargestLevelSum(self, root, k):
        """
        :type root: Optional[TreeNode]
        :type k: int
        :rtype: int
        """
        def nth_element(nums, n, left=0, compare=lambda a, b: a < b):
            def tri_partition(nums, left, right, target, compare):
                mid = left
                while mid <= right:
                    if nums[mid] == target:
                        mid += 1
                    elif compare(nums[mid], target):
                        nums[left], nums[mid] = nums[mid], nums[left]
                        left += 1
                        mid += 1
                    else:
                        nums[mid], nums[right] = nums[right], nums[mid]
                        right -= 1
                return left, right
            
            right = len(nums)-1
            while left <= right:
                pivot_idx = random.randint(left, right)
                pivot_left, pivot_right = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left-1
                else:  # pivot_right < n.
                    left = pivot_right+1
    
        arr = []
        q = [root]
        while q:
            new_q = []
            for u in q:
                if u.left:
                    new_q.append(u.left)
                if u.right:
                    new_q.append(u.right)
            arr.append(sum(x.val for x in q))
            q = new_q
        if k-1 >= len(arr):
            return -1
        nth_element(arr, k-1, compare=lambda a, b: a > b)
        return arr[k-1]

# 15 December 2023

# Time:  O(n^2)
# Space: O(n^2)

class Solution(object):
    # @param s, a string
    # @return an integer
    def minCut(self, s):
        lookup = [[False for j in xrange(len(s))] for i in xrange(len(s))]
        mincut = [len(s) - 1 - i for i in xrange(len(s) + 1)]

        for i in reversed(xrange(len(s))):
            for j in xrange(i, len(s)):
                if s[i] == s[j]  and (j - i < 2 or lookup[i + 1][j - 1]):
                    lookup[i][j] = True
                    mincut[i] = min(mincut[i], mincut[j + 1] + 1)

        return mincut[0]

# 17 December 2023

# Time:  O(n)
# Space: O(1)

# simulation, optimized from solution2
class Solution(object):
    def minMaxGame(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        while n != 1:
            new_q = []
            for i in xrange(n//2):
                nums[i] = min(nums[2*i], nums[2*i+1]) if i%2 == 0 else max(nums[2*i], nums[2*i+1])
            n //= 2
        return nums[0]


# Time:  O(n)
# Space: O(n)
# simulation
class Solution2(object):
    def minMaxGame(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        q = nums[:]
        while len(q) != 1:
            new_q = []
            for i in xrange(len(q)//2):
                new_q.append(min(q[2*i], q[2*i+1]) if i%2 == 0 else max(q[2*i], q[2*i+1]))
            q = new_q
        return q[0]

# 18 December 2023

# Time:  O(max(m, n) * min(m, n)^3)
# Space: O(m + n)

class Solution(object):
    def largestMagicSquare(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def get_sum(prefix, a, b):
            return prefix[b+1]-prefix[a]

        def check(grid, prefix_row, prefix_col, l, i, j):
            diag, anti_diag = 0, 0
            for d in xrange(l):
                diag += grid[i+d][j+d]
                anti_diag += grid[i+d][j+l-1-d]
            if diag != anti_diag:
                return False
            for ni in xrange(i, i+l):
                if diag != get_sum(prefix_row[ni], j, j+l-1):
                    return False
            for nj in xrange(j, j+l):
                if diag != get_sum(prefix_col[nj], i, i+l-1):
                    return False  
            return True

        prefix_row = [[0]*(len(grid[0])+1) for _ in xrange(len(grid))]
        prefix_col = [[0]*(len(grid)+1) for _ in xrange(len(grid[0]))]
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                prefix_row[i][j+1] = prefix_row[i][j] + grid[i][j]
                prefix_col[j][i+1] = prefix_col[j][i] + grid[i][j]
        for l in reversed(xrange(1, min(len(grid), len(grid[0]))+1)):
            for i in xrange(len(grid)-(l-1)):
                for j in xrange(len(grid[0])-(l-1)):
                    if check(grid, prefix_row, prefix_col, l, i, j):
                        return l
        return 1

# 19 December 2023

# Time:  O(6^3 * n)
# Space: O(6^2)

import collections


# dp
class Solution(object):
    def distinctSequences(self, n):
        """
        :type n: int
        :rtype: int
        """
        def gcd(a, b):
            while b:
                a, b = b, a%b
            return a

        if n == 1:
            return 6
        MOD = 10**9 + 7
        dp = [[0]*6 for _ in xrange(6)]
        for i in xrange(6):
            for j in xrange(6):
                if i != j and gcd(i+1, j+1) == 1:
                    dp[i][j] = 1
        for _ in xrange(n-2):
            new_dp = [[0]*6 for _ in xrange(6)]
            for i in xrange(6):
                for j in xrange(6):
                    if not dp[i][j]:
                        continue
                    for k in xrange(6):
                        if not dp[j][k]:
                            continue
                        if k != i:
                            new_dp[i][j] = (new_dp[i][j]+dp[j][k]) % MOD
            dp = new_dp
        return sum(dp[i][j] for i in xrange(6) for j in xrange(6)) % MOD

# 20 December 2023

# Time:  O(n)
# Space: O(h)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

        
class Solution(object):
    def flipMatchVoyage(self, root, voyage):
        """
        :type root: TreeNode
        :type voyage: List[int]
        :rtype: List[int]
        """
        def dfs(root, voyage, i, result):
            if not root:
                return True
            if root.val != voyage[i[0]]:
                return False
            i[0] += 1
            if root.left and root.left.val != voyage[i[0]]:
                result.append(root.val)
                return dfs(root.right, voyage, i, result) and \
                       dfs(root.left, voyage, i, result)
            return dfs(root.left, voyage, i, result) and \
                   dfs(root.right, voyage, i, result)
        
        result = []
        return result if dfs(root, voyage, [0], result) else [-1]

# 21 December 2023

# Time:  O(n)
# Space: O(1)

class Solution(object):
    # @param A, a list of integers
    # @return an integer
    def firstMissingPositive(self, A):
        i = 0
        while i < len(A):
            if A[i] > 0 and A[i] - 1 < len(A) and A[i] != A[A[i]-1]:
                A[A[i]-1], A[i] = A[i], A[A[i]-1]
            else:
                i += 1

        for i, integer in enumerate(A):
            if integer != i + 1:
                return i + 1
        return len(A) + 1

# 22 December 2023

# Time:  O(logn)
# Space: O(1)

# math
class Solution(object):
    def countEven(self, num):
        """
        :type num: int
        :rtype: int
        """
        def parity(x):
            result = 0
            while x:
                result += x%10
                x //= 10
            return result%2

        return (num-parity(num))//2


# Time:  O(nlogn)
# Space: O(1)
# brute force
class Solution2(object):
    def countEven(self, num):
        """
        :type num: int
        :rtype: int
        """
        def parity(x):
            result = 0
            while x:
                result += x%10
                x //= 10
            return result%2

        return sum(parity(x) == 0 for x in xrange(1, num+1))


# Time:  O(nlogn)
# Space: O(logn)
# brute force
class Solution3(object):
    def countEven(self, num):
        """
        :type num: int
        :rtype: int
        """
        return sum(sum(map(int, str(x)))%2 == 0 for x in xrange(1, num+1))

# 23 December 2023

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

