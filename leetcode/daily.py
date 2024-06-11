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

# 24 December 2023

# Time:  O(nlogs), s is the sum of nums
# Space: O(1)

class Solution(object):
    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """
        def check(nums, m, s):
            cnt, curr_sum = 1, 0
            for num in nums:
                curr_sum += num
                if curr_sum > s:
                    curr_sum = num
                    cnt += 1
            return cnt <= m

        left, right = max(nums), sum(nums)
        while left <= right:
            mid = left + (right - left) // 2
            if check(nums, m, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left

# 25 December 2023

# Time:  O(n)
# Space: O(h)

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        if d in (0, 1):
            node = TreeNode(v)
            if d == 1:
                node.left = root
            else:
                node.right = root
            return node
        if root and d >= 2:
            root.left = self.addOneRow(root.left,  v, d-1 if d > 2 else 1)
            root.right = self.addOneRow(root.right, v, d-1 if d > 2 else 0)
        return root

# 26 December 2023

# Time:  O(n^2)
# Space: O(1)

class Solution(object):
    def numTeams(self, rating):
        """
        :type rating: List[int]
        :rtype: int
        """
        result = 0
        for i in xrange(1, len(rating)-1):
            less, greater = [0]*2, [0]*2
            for j in xrange(len(rating)):
                if rating[i] > rating[j]:
                    less[i < j] += 1
                if rating[i] < rating[j]:
                    greater[i < j] += 1
            result += less[0]*greater[1] + greater[0]*less[1]
        return result

# 27 December 2023

# Time:  O(n)
# Space: O(1)

# dp
class Solution(object):
    def validPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        dp = [False]*4
        dp[0] = True
        for i in xrange(len(nums)):
            dp[(i+1)%4] = False
            if i-1 >= 0 and nums[i] == nums[i-1]:
                dp[(i+1)%4] |= dp[((i+1)-2)%4]
            if i-2 >= 0 and (nums[i] == nums[i-1] == nums[i-2] or
                             nums[i] == nums[i-1]+1 == nums[i-2]+2):
                dp[(i+1)%4] |= dp[((i+1)-3)%4]
        return dp[len(nums)%4]

# 29 December 2023

# Time:  O(n^3)
# Space: O(n^2)

# weighted bipartite matching solution
class Solution(object):
    def minimumXORSum(self, nums1, nums2):
        # Template translated from:
        # https://github.com/kth-competitive-programming/kactl/blob/main/content/graph/WeightedMatching.h
        def hungarian(a):  # Time: O(n^2 * m), Space: O(n + m)
            if not a:
                return 0, []
            n, m = len(a)+1, len(a[0])+1
            u, v, p, ans = [0]*n, [0]*m, [0]*m, [0]*(n-1)
            for i in xrange(1, n):
                p[0] = i
                j0 = 0  # add "dummy" worker 0
                dist, pre = [float("inf")]*m, [-1]*m
                done = [False]*(m+1)
                while True:  # dijkstra
                    done[j0] = True
                    i0, j1, delta = p[j0], None, float("inf")
                    for j in xrange(1, m):
                        if done[j]:
                            continue
                        cur = a[i0-1][j-1]-u[i0]-v[j]
                        if cur < dist[j]:
                            dist[j], pre[j] = cur, j0
                        if dist[j] < delta:
                            delta, j1 = dist[j], j
                    for j in xrange(m):
                        if done[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            dist[j] -= delta
                    j0 = j1
                    if not p[j0]:
                        break
                while j0:  # update alternating path
                    j1 = pre[j0]
                    p[j0], j0 = p[j1], j1
            for j in xrange(1, m):
                if p[j]:
                    ans[p[j]-1] = j-1
            return -v[0], ans  # min cost
        
        adj = [[0]*len(nums2) for _ in xrange(len(nums1))]
        for i in xrange(len(nums1)):
            for j in xrange(len(nums2)):
                adj[i][j] = nums1[i]^nums2[j]
        return hungarian(adj)[0]


# Time:  O(n * 2^n)
# Space: O(2^n)
# dp solution
class Solution2(object):
    def minimumXORSum(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        dp = [(float("inf"), float("inf"))]*(2**len(nums2))
        dp[0] = (0, 0)
        for mask in xrange(len(dp)):
            bit = 1
            for i in xrange(len(nums2)):
                if (mask&bit) == 0:
                    dp[mask|bit] = min(dp[mask|bit], (dp[mask][0]+(nums1[dp[mask][1]]^nums2[i]), dp[mask][1]+1))
                bit <<= 1
        return dp[-1][0]

# 30 December 2023

# Time:  O(n)
# Space: O(1)

# Rabin-Karp Algorithm
class Solution(object):
    def longestDecomposition(self, text):
        """
        :type text: str
        :rtype: int
        """
        def compare(text, l, s1, s2):
            for i in xrange(l):
                if text[s1+i] != text[s2+i]:
                    return False
            return True

        MOD = 10**9+7
        D = 26
        result = 0
        left, right, l, pow_D = 0, 0, 0, 1
        for i in xrange(len(text)):
            left = (D*left + (ord(text[i])-ord('a'))) % MOD
            right = (pow_D*(ord(text[-1-i])-ord('a')) + right) % MOD
            l += 1
            pow_D = (pow_D*D) % MOD 
            if left == right and compare(text, l, i-l+1, len(text)-1-i):
                result += 1
                left, right, l, pow_D = 0, 0, 0, 1
        return result

# 31 December 2023

# Time:  O(n)
# Space: O(n)

import collections


class Solution(object):
    def waysToPartition(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        total = sum(nums)
        right = collections.Counter()
        prefix = 0
        for i in xrange(len(nums)-1):
            prefix += nums[i]
            right[prefix-(total-prefix)] += 1
        result = right[0]
        left = collections.Counter()
        prefix = 0
        for x in nums:
            result = max(result, left[k-x]+right[-(k-x)])
            prefix += x
            left[prefix-(total-prefix)] += 1
            right[prefix-(total-prefix)] -= 1
        return result

# 01 January 2024

# Time:  O(n + logm), n is the length of text, m is the number of fonts
# Space: O(1)

import collections


class FontInfo(object):
    def getWidth(self, fontSize, ch):
        """
        :type fontSize: int
        :type ch: char
        :rtype int
        """
        pass
    
    def getHeight(self, fontSize):
        """
        :type fontSize: int
        :rtype int
        """
        pass


class Solution(object):
    def maxFont(self, text, w, h, fonts, fontInfo):
        """
        :type text: str
        :type w: int
        :type h: int
        :type fonts: List[int]
        :type fontInfo: FontInfo
        :rtype: int
        """
        def check(count, w, h, fonts, fontInfo, x):  # Time: O(1)
            return (fontInfo.getHeight(fonts[x]) <= h and
                    sum(cnt * fontInfo.getWidth(fonts[x], c) for c, cnt in count.iteritems()) <= w)

        count = collections.Counter(text)
        left, right = 0, len(fonts)-1
        while left <= right:
            mid = left + (right-left)//2
            if not check(count, w, h, fonts, fontInfo, mid):
                right = mid-1
            else:
                left = mid+1
        return fonts[right] if right >= 0 else -1

# 02 January 2024

# Time:  O(n)
# Space: O(k)

# constructive algorithms
class Solution(object):
    def shortestSequence(self, rolls, k):
        """
        :type rolls: List[int]
        :type k: int
        :rtype: int
        """
        l = 0
        lookup = set()
        for x in rolls:
            lookup.add(x)
            if len(lookup) != k:
                continue
            lookup.clear()
            l += 1
        return l+1

# 03 January 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    def minimumBuckets(self, street):
        """
        :type street: str
        :rtype: int
        """
        result = 0
        street = list(street)
        for i, c in enumerate(street):
            if c != 'H' or (i and street[i-1] == 'B'):
                continue
            if i+1 < len(street) and street[i+1] == '.':
                street[i+1] = 'B'
                result += 1
            elif i and street[i-1] == '.':
                street[i-1] = 'B'
                result += 1
            else:
                return -1
        return result

# 04 January 2024

# Time:  O(n)
# Space: O(h)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def getAllElements(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: List[int]
        """
        def inorder_gen(root):
            result, stack = [], [(root, False)]
            while stack:
                root, is_visited = stack.pop()
                if root is None:
                    continue
                if is_visited:
                    yield root.val
                else:
                    stack.append((root.right, False))
                    stack.append((root, True))
                    stack.append((root.left, False))
            yield None
        
        result = []
        left_gen, right_gen = inorder_gen(root1), inorder_gen(root2)
        left, right = next(left_gen), next(right_gen)
        while left is not None or right is not None:
            if right is None or (left is not None and left < right):
                result.append(left)
                left = next(left_gen)
            else:
                result.append(right)
                right = next(right_gen)
        return result
  

# 05 January 2024

# Time:  O(n^3)
# Space: O(n^2)

class Solution(object):
    def largest1BorderedSquare(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        top, left = [a[:] for a in grid], [a[:] for a in grid]
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if not grid[i][j]:
                    continue
                if i:
                    top[i][j] = top[i-1][j] + 1
                if j:
                    left[i][j] = left[i][j-1] + 1
        for l in reversed(xrange(1, min(len(grid), len(grid[0]))+1)):
            for i in xrange(len(grid)-l+1):
                for j in xrange(len(grid[0])-l+1):
                    if min(top[i+l-1][j],
                           top[i+l-1][j+l-1],
                           left[i][j+l-1],
                           left[i+l-1][j+l-1]) >= l:
                        return l*l
        return 0

# 06 January 2024

# Time:  O(klogn)
# Space: O(1)

class Solution(object):
    def superEggDrop(self, K, N):
        """
        :type K: int
        :type N: int
        :rtype: int
        """
        def check(n, K, N):
            # let f(n, K) be the max number of floors could be solved by n moves and K eggs,
            # we want to do binary search to find min of n, s.t. f(n, K) >= N,
            # if we use one move to drop egg with X floors
            # 1. if it breaks, we can search new X in the range [X+1, X+f(n-1, K-1)]
            # 2. if it doesn't break, we can search new X in the range [X-f(n-1, K), X-1]
            # => f(n, K) = (X+f(n-1, K-1))-(X-f(n-1, K))+1 = f(n-1, K-1)+f(n-1, K)+1
            # => (1) f(n, K)   = f(n-1, K)  +1+f(n-1, K-1)
            #    (2) f(n, K-1) = f(n-1, K-1)+1+f(n-1, K-2)
            # let g(n, K) = f(n, K)-f(n, K-1), and we subtract (1) by (2)
            # => g(n, K) = g(n-1, K)+g(n-1, K-1), obviously, it is binomial coefficient
            # => C(n, K) = g(n, K) = f(n, K)-f(n, K-1),
            #    which also implies if we have one more egg with n moves and x-1 egges, we can have more C(n, x) floors solvable
            # => f(n, K) = C(n, K)+f(n, K-1) = C(n, K) + C(n, K-1) + ... + C(n, 1) + f(n, 0) = sum(C(n, k) for k in [1, K])
            # => all we have to do is to check sum(C(n, k) for k in [1, K]) >= N,
            #    if true, there must exist a 1-to-1 mapping from each F in [1, N] to each sucess and failure sequence of every C(n, k) combinations for k in [1, K]
            total, c = 0, 1
            for k in xrange(1, K+1):
                c *= n-k+1
                c //= k
                total += c
                if total >= N:
                    return True
            return False

        left, right = 1, N
        while left <= right:
            mid = left + (right-left)//2
            if check(mid, K, N):
                right = mid-1
            else:
                left = mid+1
        return left

# 07 January 2024

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

# 07 January 2024

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
# 08 January 2024

# Time:  O(26 + d * n), d = len(set(word))
# Space: O(26)

# freq table, two pointers, sliding window
class Solution(object):
    def countCompleteSubstrings(self, word, k):
        """
        :type word: str
        :type k: int
        :rtype: int
        """
        result = valid = 0
        cnt = [0]*26
        for c in xrange(1, len(set(word))+1):
            left = 0
            for right in xrange(len(word)):
                cnt[ord(word[right])-ord('a')] += 1
                curr = cnt[ord(word[right])-ord('a')]
                valid += 1 if curr == k else -1 if curr == k+1 else 0
                if right-left+1 == c*k+1:
                    curr = cnt[ord(word[left])-ord('a')]
                    valid -= 1 if curr == k else -1 if curr == k+1 else 0
                    cnt[ord(word[left])-ord('a')] -= 1
                    left += 1
                if valid == c:
                    result += 1
                if right+1 == len(word) or abs(ord(word[right+1])-ord(word[right])) > 2:
                    while left < right+1:
                        curr = cnt[ord(word[left])-ord('a')]
                        valid -= 1 if curr == k else -1 if curr == k+1 else 0
                        cnt[ord(word[left])-ord('a')] -= 1
                        left += 1
        return result

# 09 January 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    def minTaps(self, n, ranges):
        """
        :type n: int
        :type ranges: List[int]
        :rtype: int
        """
        def jump_game(A):
            jump_count, reachable, curr_reachable = 0, 0, 0
            for i, length in enumerate(A):
                if i > reachable:
                    return -1
                if i > curr_reachable:
                    curr_reachable = reachable
                    jump_count += 1
                reachable = max(reachable, i+length)
            return jump_count
    
        max_range = [0]*(n+1)
        for i, r in enumerate(ranges):
            left, right = max(i-r, 0), min(i+r, n)
            max_range[left] = max(max_range[left], right-left)
        return jump_game(max_range)

# 11 January 2024

# Time:  O(nlogn) for total n addNums, O(logn) per addNum, O(1) per findMedian.
# Space: O(n), total space

from heapq import heappush, heappop

class MedianFinder(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.__max_heap = []
        self.__min_heap = []

    def addNum(self, num):
        """
        Adds a num into the data structure.
        :type num: int
        :rtype: void
        """
        # Balance smaller half and larger half.
        if not self.__max_heap or num > -self.__max_heap[0]:
            heappush(self.__min_heap, num)
            if len(self.__min_heap) > len(self.__max_heap) + 1:
                heappush(self.__max_heap, -heappop(self.__min_heap))
        else:
            heappush(self.__max_heap, -num)
            if len(self.__max_heap) > len(self.__min_heap):
                heappush(self.__min_heap, -heappop(self.__max_heap))

    def findMedian(self):
        """
        Returns the median of current data stream
        :rtype: float
        """
        return (-self.__max_heap[0] + self.__min_heap[0]) / 2.0 \
               if len(self.__min_heap) == len(self.__max_heap) \
               else self.__min_heap[0]

# 12 January 2024

# Time:  O((m * n) * (m + n) / 32)
# Space: O(n * (m + n) / 32)

# dp with bitsets
class Solution(object):
    def hasValidPath(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: bool
        """
        if (len(grid)+len(grid[0])-1)%2:
            return False
        dp = [0]*(len(grid[0])+1)
        for i in xrange(len(grid)):
            dp[0] = int(not i)
            for j in xrange(len(grid[0])):
                dp[j+1] = (dp[j]|dp[j+1])<<1 if grid[i][j] == '(' else (dp[j]|dp[j+1])>>1
        return dp[-1]&1


# Time:  O(m * n)
# Space: O(n)
# dp, optimized from solution1 (wrong answer)
class Solution_WA(object):
    def hasValidPath(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: bool
        """
        if (len(grid)+len(grid[0])-1)%2:
            return False
        dp = [[float("inf"), float("-inf")] for _ in xrange(len(grid[0])+1)]
        for i in xrange(len(grid)):
            dp[0] = [0, 0] if not i else [float("inf"), float("-inf")]
            for j in xrange(len(grid[0])):
                d = 1 if grid[i][j] == '(' else -1
                dp[j+1] = [min(dp[j+1][0], dp[j][0])+d, max(dp[j+1][1], dp[j][1])+d]
                # bitset pattern is like xxx1010101xxxx (in fact, it is not always true in this problem where some paths are invalid)
                if dp[j+1][1] < 0:
                    dp[j+1] = [float("inf"), float("-inf")]
                else:
                    dp[j+1][0] = max(dp[j+1][0], dp[j+1][1]%2)
        return dp[-1][0] == 0

# 13 January 2024

# Time:  O(n)
# Space: O(1)

import itertools


# greedy, kadane's algorithm
class Solution(object):
    def maximumCostSubstring(self, s, chars, vals):
        """
        :type s: str
        :type chars: str
        :type vals: List[int]
        :rtype: int
        """
        def kadane(s):
            result = curr = 0
            for c in s:
                curr = max(curr+(lookup[c] if c in lookup else ord(c)-ord('a')+1), 0)
                result = max(result, curr)
            return result

        lookup = {}
        for c, v in itertools.izip(chars, vals):
            lookup[c] = v
        return kadane(s)

# 14 January 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def firstPalindrome(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        def is_palindrome(s):
            i, j = 0, len(s)-1
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        for w in words:
            if is_palindrome(w):
                return w
        return ""

 
# Time:  O(n)
# Space: O(l), l is the max length of words
class Solution2(object):
    def firstPalindrome(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        return next((x for x in words if x == x[::-1]), "")

# 15 January 2024

# Time:  O(n^2 * m), m is the length of targetPath
# Space: O(n * m)

class Solution(object):
    def mostSimilar(self, n, roads, names, targetPath):
        """
        :type n: int
        :type roads: List[List[int]]
        :type names: List[str]
        :type targetPath: List[str]
        :rtype: List[int]
        """
        adj = [[] for _ in xrange(n)]
        for u, v in roads:
            adj[u].append(v)
            adj[v].append(u)

        dp = [[0]*n for _ in xrange(len(targetPath)+1)]
        for i in xrange(1, len(targetPath)+1):
            for v in xrange(n):
                dp[i][v] = (names[v] != targetPath[i-1]) + min(dp[i-1][u] for u in adj[v]) 

        path = [dp[-1].index(min(dp[-1]))]
        for i in reversed(xrange(2, len(targetPath)+1)):
            for u in adj[path[-1]]:
                if dp[i-1][u]+(names[path[-1]] != targetPath[i-1]) == dp[i][path[-1]]:
                    path.append(u)
                    break
        return path[::-1]

# 16 January 2024

# Time:  O(n)
# Space: O(1)

# greedy, combinatorics
class Solution(object):
    def numberOfWays(self, corridor):
        """
        :type corridor: str
        :rtype: int
        """
        MOD = 10**9+7
        result, cnt, j = 1, 0, -1
        for i, x in enumerate(corridor):
            if x != 'S':
                continue
            cnt += 1
            if cnt >= 3 and cnt%2:
                result = result*(i-j)%MOD
            j = i
        return result if cnt and cnt%2 == 0 else 0

# 18 January 2024

# Time:  O(n)
# Space: O(1)

# string
class Solution(object):
    def isCircularSentence(self, sentence):
        """
        :type sentence: str
        :rtype: bool
        """
        return sentence[0] == sentence[-1] and all(sentence[i-1] == sentence[i+1]for i in xrange(len(sentence)) if sentence[i] == ' ')

# 19 January 2024

# Time:  O(n)
# Space: O(h)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        UNCOVERED, COVERED, CAMERA = range(3)
        def dfs(root, result):
            left = dfs(root.left, result) if root.left else COVERED
            right = dfs(root.right, result) if root.right else COVERED
            if left == UNCOVERED or right == UNCOVERED:
                result[0] += 1
                return CAMERA
            if left == CAMERA or right == CAMERA:
                return COVERED
            return UNCOVERED
        
        result = [0]
        if dfs(root, result) == UNCOVERED:
            result[0] += 1
        return result[0]

# 21 January 2024

# Time:  O(nlogn + nlogr), r is the range of positions
# Space: O(1)

class Solution(object):
    def maxDistance(self, position, m):
        """
        :type position: List[int]
        :type m: int
        :rtype: int
        """
        def check(position, m, x):
            count, prev = 1, position[0]
            for i in xrange(1, len(position)):
                if position[i]-prev >= x:
                    count += 1
                    prev = position[i]
            return count >= m
        
        position.sort()
        left, right = 1, position[-1]-position[0]
        while left <= right:
            mid = left + (right-left)//2
            if not check(position, m, mid):
                right = mid-1
            else:
                left = mid+1
        return right

# 22 January 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    def canFormArray(self, arr, pieces):
        """
        :type arr: List[int]
        :type pieces: List[List[int]]
        :rtype: bool
        """
        lookup = {x[0]: i for i, x in enumerate(pieces)}
        i = 0
        while i < len(arr): 
            if arr[i] not in lookup:
                return False
            for c in pieces[lookup[arr[i]]]:
                if i == len(arr) or arr[i] != c:
                    return False
                i += 1
        return True 

# 23 January 2024

# Time:  O(h)
# Space: O(1)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def insertIntoMaxTree(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if not root:
            return TreeNode(val)

        if val > root.val:
            node = TreeNode(val)
            node.left = root
            return node
        
        curr = root
        while curr.right and curr.right.val > val:
            curr = curr.right
        node = TreeNode(val)
        curr.right, node.left = node, curr.right
        return root

# 24 January 2024

# Time:  O(m * n)
# Space: O(m + n)

# array
class Solution(object):
    def onesMinusZeros(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: List[List[int]]
        """
        rows = [sum(grid[i][j] for j in xrange(len(grid[0]))) for i in xrange(len(grid))]
        cols = [sum(grid[i][j] for i in xrange(len(grid))) for j in xrange(len(grid[0]))]
        return [[rows[i]+cols[j]-(len(grid)-rows[i])-(len(grid[0])-cols[j]) for j in xrange(len(grid[0]))] for i in xrange(len(grid))]

# 25 January 2024

# Time:  O(n)
# Space: O(n)

import itertools


class Solution(object):
    def flipgame(self, fronts, backs):
        """
        :type fronts: List[int]
        :type backs: List[int]
        :rtype: int
        """
        same = {n for i, n in enumerate(fronts) if n == backs[i]}
        result = float("inf")
        for n in itertools.chain(fronts, backs):
            if n not in same:
                result = min(result, n)
        return result if result < float("inf") else 0

# 26 January 2024

# Time:  O(1)
# Space: O(1)

class Solution(object):
    def convertToBase7(self, num):
        if num < 0:
            return '-' + self.convertToBase7(-num)
        result = ''
        while num:
            result = str(num % 7) + result
            num //= 7
        return result if result else '0'


class Solution2(object):
    def convertToBase7(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num < 0:
            return '-' + self.convertToBase7(-num)
        if num < 7:
            return str(num)
        return self.convertToBase7(num // 7) + str(num % 7)

# 27 January 2024

# Time:  O(1)
# Space: O(1)

# math
class Solution(object):
    def findDelayedArrivalTime(self, arrivalTime, delayedTime):
        """
        :type arrivalTime: int
        :type delayedTime: int
        :rtype: int
        """
        return (arrivalTime + delayedTime)%24

# 28 January 2024

# Time:  O(n)
# Space: O(1)

import itertools


class Solution(object):
    def gridGame(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        result = float("inf")
        left, right = 0, sum(grid[0])
        for a, b in itertools.izip(grid[0], grid[1]):
            right -= a
            result = min(result, max(left, right))
            left += b
        return result

# 29 January 2024

# Time:  O(n)
# Space: O(n)

import collections
import operator


# combinatorics, dp
class Solution(object):
    def countTheNumOfKFreeSubsets(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def count(x):
            y = x
            while y-k in cnt:
                y -= k
            dp = [1, 0]  # dp[0]: count without i, dp[1]: count with i
            for i in xrange(y, x+1, k):
                dp = [dp[0]+dp[1], dp[0]*((1<<cnt[i])-1)]
            return sum(dp)

        cnt = collections.Counter(nums)
        return reduce(operator.mul, (count(i) for i in cnt.iterkeys() if i+k not in cnt))

# 30 January 2024

# Time:  O(n^3 / k)
# Space: O(n^2)

class Solution(object):
    def mergeStones(self, stones, K):
        """
        :type stones: List[int]
        :type K: int
        :rtype: int
        """
        if (len(stones)-1) % (K-1):
            return -1
        prefix = [0]
        for x in stones:
            prefix.append(prefix[-1]+x)
        dp = [[0]*len(stones) for _ in xrange(len(stones))]
        for l in xrange(K-1, len(stones)):
            for i in xrange(len(stones)-l):
                dp[i][i+l] = min(dp[i][j]+dp[j+1][i+l] for j in xrange(i, i+l, K-1))
                if l % (K-1) == 0:
                    dp[i][i+l] += prefix[i+l+1] - prefix[i]
        return dp[0][len(stones)-1]

# 31 January 2024

# Time:  O(n)
# Space: O(1)

# prefix sum
class Solution(object):
    def findIndices(self, nums, indexDifference, valueDifference):
        """
        :type nums: List[int]
        :type indexDifference: int
        :type valueDifference: int
        :rtype: List[int]
        """
        mx_i = mn_i = 0
        for i in xrange(len(nums)-indexDifference):
            if nums[i] > nums[mx_i]:
                mx_i = i
            elif nums[i] < nums[mn_i]:
                mn_i = i
            # we don't need to add abs for the difference since
            # - if nums[mx_i]-nums[i+indexDifference] < 0, then checking nums[i+indexDifference]-nums[mn_i] >= -(nums[mx_i]-nums[i+indexDifference]) > 0 can cover the case
            # - if nums[i+indexDifference]-nums[mn_i] < 0, then checking nums[mx_i]-nums[i+indexDifference] >= -(nums[i+indexDifference]-nums[mn_i]) > 0 can cover the case
            if nums[mx_i]-nums[i+indexDifference] >= valueDifference:
                return [mx_i, i+indexDifference]
            if nums[i+indexDifference]-nums[mn_i] >= valueDifference:
                return [mn_i, i+indexDifference]
        return [-1]*2

# 01 February 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    # @param ratings, a list of integer
    # @return an integer
    def candy(self, ratings):
        candies = [1 for _ in xrange(len(ratings))]
        for i in xrange(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1

        for i in reversed(xrange(1, len(ratings))):
            if ratings[i - 1] > ratings[i] and candies[i - 1] <= candies[i]:
                candies[i - 1] = candies[i] + 1

        return sum(candies)

# 02 February 2024

# Time:  O(mlogm + nlogn + mlogn)
# Space: O(1)

import bisect


class Solution(object):
    def minWastedSpace(self, packages, boxes):
        """
        :type packages: List[int]
        :type boxes: List[List[int]]
        :rtype: int
        """
        MOD = 10**9+7
        INF = float("inf")

        packages.sort()
        result = INF
        for box in boxes:
            box.sort()
            if box[-1] < packages[-1]:
                continue
            curr = left = 0
            for b in box:
                right = bisect.bisect_right(packages, b, left)
                curr += b * (right-left)
                left = right
            result = min(result, curr)
        return (result-sum(packages))%MOD if result != INF else -1

# 03 February 2024

# Time:  O(n * r^2)
# Space: O(min(n * r^2, max_x * max_y))

# math, hash table
class Solution(object):
    def countLatticePoints(self, circles):
        """
        :type circles: List[List[int]]
        :rtype: int
        """
        lookup = set()
        for x, y, r in circles:
            for i in xrange(-r, r+1):
                for j in xrange(-r, r+1):
                    if i**2+j**2 <= r**2:
                        lookup.add(((x+i), (y+j)))
        return len(lookup)


# Time:  O(n * max_x * max_y)
# Space: O(1)
# math
class Solution2(object):
    def countLatticePoints(self, circles):
        """
        :type circles: List[List[int]]
        :rtype: int
        """
        max_x = max(x+r for x, _, r in circles)
        max_y = max(y+r for _, y, r in circles)
        result = 0
        for i in xrange(max_x+1):
            for j in xrange(max_y+1):
                if any((i-x)**2+(j-y)**2 <= r**2 for x, y, r in circles):
                    result += 1
        return result

# 04 February 2024

# Time:  addNum: O(n), getIntervals: O(n), n is the number of disjoint intervals.
# Space: O(n)

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


class SummaryRanges(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.__intervals = []

    def addNum(self, val):
        """
        :type val: int
        :rtype: void
        """
        def upper_bound(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) / 2
                if nums[mid].start > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        i = upper_bound(self.__intervals, val)
        start, end = val, val
        if i != 0 and self.__intervals[i-1].end + 1 >= val:
            i -= 1
        while i != len(self.__intervals) and \
              end + 1 >= self.__intervals[i].start:
            start = min(start, self.__intervals[i].start)
            end = max(end, self.__intervals[i].end)
            del self.__intervals[i]
        self.__intervals.insert(i, Interval(start, end))

    def getIntervals(self):
        """
        :rtype: List[Interval]
        """
        return self.__intervals

# 05 February 2024

# Time:  O(m * n)
# Space: O(g)

from collections import deque

class Solution(object):
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: void Do not return anything, modify rooms in-place instead.
        """
        INF = 2147483647
        q = deque([(i, j) for i, row in enumerate(rooms) for j, r in enumerate(row) if not r])
        while q:
            (i, j) = q.popleft()
            for I, J in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
                if 0 <= I < len(rooms) and 0 <= J < len(rooms[0]) and \
                   rooms[I][J] == INF:
                    rooms[I][J] = rooms[i][j] + 1
                    q.append((I, J))

# 06 February 2024

# Time:  O(n + q)
# Space: O(n)

class Solution(object):
    def platesBetweenCandles(self, s, queries):
        """
        :type s: str
        :type queries: List[List[int]]
        :rtype: List[int]
        """
        left, prefix = [0]*len(s), {}
        curr, cnt = -1, 0
        for i in xrange(len(s)):
            if s[i] == '|':
                curr = i
                cnt += 1
                prefix[i] = cnt
            left[i] = curr
        right = [0]*len(s)
        curr = len(s)
        for i in reversed(xrange(len(s))):
            if s[i] == '|':
                curr = i
            right[i] = curr
        return [max((left[r]-right[l]+1) - (prefix[left[r]]-prefix[right[l]]+1), 0) for l, r in queries]

# 07 February 2024

# Time:  O(n)
# Space: O(h)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def maximumAverageSubtree(self, root):
        """
        :type root: TreeNode
        :rtype: float
        """
        def maximumAverageSubtreeHelper(root, result):
            if not root:
                return [0.0, 0]
            s1, n1 = maximumAverageSubtreeHelper(root.left, result)
            s2, n2 = maximumAverageSubtreeHelper(root.right, result)
            s = s1+s2+root.val
            n = n1+n2+1
            result[0] = max(result[0], s / n)
            return [s, n]

        result = [0]
        maximumAverageSubtreeHelper(root, result)
        return result[0]

# 08 February 2024

# Time:  O(n)
# Space: O(k)

import collections


class Solution(object):
    def maxResult(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        score = 0
        dq = collections.deque()
        for i, num in enumerate(nums):
            if dq and dq[0][0] == i-k-1:
                dq.popleft()
            score = num if not dq else dq[0][1]+num
            while dq and dq[-1][1] <= score:
                dq.pop()
            dq.append((i, score))
        return score

# 09 February 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def isMonotonic(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        inc, dec = False, False
        for i in xrange(len(A)-1):
            if A[i] < A[i+1]:
                inc = True
            elif A[i] > A[i+1]:
                dec = True
        return not inc or not dec

# 10 February 2024

# Time:  O(n)
# Space: O(n)

import collections


class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        result = 0
        accumulated_sum = 0
        lookup = collections.defaultdict(int)
        lookup[0] += 1
        for num in nums:
            accumulated_sum += num
            result += lookup[accumulated_sum - k]
            lookup[accumulated_sum] += 1
        return result

# 11 February 2024

# Time:  O(nlogn)
# Space: O(n)

class Solution(object):
    def arrangeWords(self, text):
        """
        :type text: str
        :rtype: str
        """
        result = text.split()
        result[0] = result[0].lower()
        result.sort(key=len) 
        result[0] = result[0].title()
        return " ".join(result)

# 12 February 2024

# Time:  push:    O(1)
#        pop:     O(n), there is no built-in SortedDict in python. If applied, it could be reduced to O(logn)
#        popMax:  O(n)
#        top:     O(1)
#        peekMax: O(1)
# Space: O(n), n is the number of values in the current stack

import collections


class MaxStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.__idx_to_val = collections.defaultdict(int)
        self.__val_to_idxs = collections.defaultdict(list)
        self.__top = None
        self.__max = None


    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        idx = self.__val_to_idxs[self.__top][-1]+1 if self.__val_to_idxs else 0
        self.__idx_to_val[idx] = x
        self.__val_to_idxs[x].append(idx)
        self.__top = x
        self.__max = max(self.__max, x)


    def pop(self):
        """
        :rtype: int
        """
        val = self.__top
        self.__remove(val)
        return val


    def top(self):
        """
        :rtype: int
        """
        return self.__top


    def peekMax(self):
        """
        :rtype: int
        """
        return self.__max


    def popMax(self):
        """
        :rtype: int
        """
        val = self.__max
        self.__remove(val)
        return val


    def __remove(self, val):
        idx = self.__val_to_idxs[val][-1]
        self.__val_to_idxs[val].pop()
        if not self.__val_to_idxs[val]:
            del self.__val_to_idxs[val]
        del self.__idx_to_val[idx]
        if val == self.__top:
            self.__top = self.__idx_to_val[max(self.__idx_to_val.keys())] if self.__idx_to_val else None
        if val == self.__max:
            self.__max = max(self.__val_to_idxs.keys()) if self.__val_to_idxs else None

# 13 February 2024

# Time:  O(n)
# Space: O(h)

class Node(object):
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right


class Solution(object):
    def treeToDoublyList(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return None
        left_head, left_tail, right_head, right_tail = root, root, root, root
        if root.left:
            left_head = self.treeToDoublyList(root.left)
            left_tail = left_head.left
        if root.right:
            right_head = self.treeToDoublyList(root.right)
            right_tail = right_head.left
        left_tail.right, right_head.left = root, root
        root.left, root.right = left_tail, right_head
        left_head.left, right_tail.right = right_tail, left_head
        return left_head

# 14 February 2024

# Time:  O((n * C(c, min(c, k))) * 2^n)
# Space: O(2^n)

import itertools


class Solution(object):
    def minNumberOfSemesters(self, n, dependencies, k):
        """
        :type n: int
        :type dependencies: List[List[int]]
        :type k: int
        :rtype: int
        """
        reqs = [0]*n
        for u, v in dependencies:
            reqs[v-1] |= 1 << (u-1)
        dp = [n]*(1<<n)
        dp[0] = 0
        for mask in xrange(1<<n):
            candidates = []
            for v in xrange(n):
                if (mask&(1<<v)) == 0 and (mask&reqs[v]) == reqs[v]:
                    candidates.append(v)
            for choice in itertools.combinations(candidates, min(len(candidates), k)):
                new_mask = mask
                for v in choice:
                    new_mask |= 1<<v
                dp[new_mask] = min(dp[new_mask], dp[mask]+1)
        return dp[-1]


# Time:  O(nlogn + e), e is the number of edges in graph
# Space: O(n + e)
import collections
import heapq

# wrong greedy solution
# since the priority of courses are hard to decide especially for those courses with zero indegrees are of the same outdegrees and depths
# e.x.
# 9
# [[1,4],[1,5],[3,5],[3,6],[2,6],[2,7],[8,4],[8,5],[9,6],[9,7]]
# 3
class Solution_WA(object):
    def minNumberOfSemesters(self, n, dependencies, k):
        """
        :type n: int
        :type dependencies: List[List[int]]
        :type k: int
        :rtype: int
        """
        def dfs(graph, i, depths):
            if depths[i] == -1:
                depths[i] = max(dfs(graph, child, depths) for child in graph[i])+1 if i in graph else 1
            return depths[i]
            
        degrees = [0]*n
        graph = collections.defaultdict(list)
        for u, v in dependencies:
            graph[u-1].append(v-1)
            degrees[v-1] += 1
        depths = [-1]*n
        for i in xrange(n):
            dfs(graph, i, depths)
        max_heap = []
        for i in xrange(n):
            if not degrees[i]:
                heapq.heappush(max_heap, (-depths[i], i))
        result = 0
        while max_heap:
            new_q = []
            for _ in xrange(min(len(max_heap), k)):
                _, node = heapq.heappop(max_heap)
                if node not in graph:
                    continue
                for child in graph[node]:
                    degrees[child] -= 1
                    if not degrees[child]:
                        new_q.append(child)
            result += 1
            for node in new_q:
                heapq.heappush(max_heap, (-depths[node], node))
        return result

# 15 February 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        def preProcess(s):
            if not s:
                return ['^', '$']
            T = ['^']
            for c in s:
                T +=  ['#', c]
            T += ['#', '$']
            return T

        T = preProcess(s)
        P = [0] * len(T)
        center, right = 0, 0
        for i in xrange(1, len(T) - 1):
            i_mirror = 2 * center - i
            if right > i:
                P[i] = min(right - i, P[i_mirror])
            else:
                P[i] = 0

            while T[i + 1 + P[i]] == T[i - 1 - P[i]]:
                P[i] += 1

            if i + P[i] > right:
                center, right = i, i + P[i]

        max_i = 0
        for i in xrange(1, len(T) - 1):
            if P[i] > P[max_i]:
                max_i = i
        start = (max_i - 1 - P[max_i]) // 2
        return s[start : start + P[max_i]]


# Time:  O(n^2)
# Space: O(1)
class Solution2(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        def expand(s, left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return (right-left+1)-2
        
        left, right = -1, -2
        for i in xrange(len(s)):
            l = max(expand(s, i, i), expand(s, i, i+1))
            if l > right-left+1:
                right = i+l//2
                left = right-l+1
        return s[left:right+1] if left >= 0 else ""

# 16 February 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    def checkIfExist(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        lookup = set()
        for x in arr:
            if 2*x in lookup or \
               (x%2 == 0 and x//2 in lookup):
                return True
            lookup.add(x)
        return False

# 17 February 2024

# Time:  O(n)
# Space: O(h)

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec(object):

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        def serializeHelper(node):
            if not node:
                vals.append('#')
                return
            vals.append(str(node.val))
            serializeHelper(node.left)
            serializeHelper(node.right)
        vals = []
        serializeHelper(root)
        return ' '.join(vals)


    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        def deserializeHelper():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = deserializeHelper()
            node.right = deserializeHelper()
            return node
        def isplit(source, sep):
            sepsize = len(sep)
            start = 0
            while True:
                idx = source.find(sep, start)
                if idx == -1:
                    yield source[start:]
                    return
                yield source[start:idx]
                start = idx + sepsize
        vals = iter(isplit(data, ' '))
        return deserializeHelper()


# time: O(n)
# space: O(n)

class Codec2(object):

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def gen_preorder(node):
            if not node:
                yield '#'
            else:
                yield str(node.val)
                for n in gen_preorder(node.left):
                    yield n
                for n in gen_preorder(node.right):
                    yield n
                
        return ' '.join(gen_preorder(root))
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def builder(chunk_iter):
            val = next(chunk_iter)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = builder(chunk_iter)
            node.right = builder(chunk_iter)
            return node
        
        # https://stackoverflow.com/a/42373311/568901
        chunk_iter = iter(data.split())
        return builder(chunk_iter)

# 18 February 2024

# Time:  O(n)
# Space: O(1)

# Rabin-Karp Algorithm
class Solution(object):
    def longestDecomposition(self, text):
        """
        :type text: str
        :rtype: int
        """
        def compare(text, l, s1, s2):
            for i in xrange(l):
                if text[s1+i] != text[s2+i]:
                    return False
            return True

        MOD = 10**9+7
        D = 26
        result = 0
        left, right, l, pow_D = 0, 0, 0, 1
        for i in xrange(len(text)):
            left = (D*left + (ord(text[i])-ord('a'))) % MOD
            right = (pow_D*(ord(text[-1-i])-ord('a')) + right) % MOD
            l += 1
            pow_D = (pow_D*D) % MOD 
            if left == right and compare(text, l, i-l+1, len(text)-1-i):
                result += 1
                left, right, l, pow_D = 0, 0, 0, 1
        return result

# 19 February 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def minFlips(self, target):
        """
        :type target: str
        :rtype: int
        """
        result, curr = 0, '0'
        for c in target:
            if c == curr:
                continue
            curr = c
            result += 1
        return result

# 20 February 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def breakPalindrome(self, palindrome):
        """
        :type palindrome: str
        :rtype: str
        """
        for i in xrange(len(palindrome)//2):
            if palindrome[i] != 'a':
                return palindrome[:i] + 'a' + palindrome[i+1:]
        return palindrome[:-1] + 'b' if len(palindrome) >= 2 else ""

# 21 February 2024

# Time:  O(h + v), h = len(hBars), v = len(vBars)
# Space: O(h + v)

# array, hash table
class Solution(object):
    def maximizeSquareHoleArea(self, n, m, hBars, vBars):
        """
        :type n: int
        :type m: int
        :type hBars: List[int]
        :type vBars: List[int]
        :rtype: int
        """
        def max_gap(arr):
            result = l = 1
            lookup = set(arr)
            while lookup:
                x = next(iter(lookup))
                left = x
                while left-1 in lookup:
                    left -= 1
                right = x
                while right+1 in lookup:
                    right += 1
                for i in xrange(left, right+1):
                    lookup.remove(i)
                result = max(result, (right-left+1)+1)
            return result

        return min(max_gap(hBars), max_gap(vBars))**2


# Time:  O(hlogh + vlogv), h = len(hBars), v = len(vBars)
# Space: O(1)
# array, sort
class Solution2(object):
    def maximizeSquareHoleArea(self, n, m, hBars, vBars):
        """
        :type n: int
        :type m: int
        :type hBars: List[int]
        :type vBars: List[int]
        :rtype: int
        """
        def max_gap(arr):
            arr.sort()
            result = l = 1
            for i in xrange(len(arr)):
                l += 1
                result = max(result, l)
                if i+1 != len(arr) and arr[i+1] != arr[i]+1:
                    l = 1
            return result

        return min(max_gap(hBars), max_gap(vBars))**2

# 22 February 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def alphabetBoardPath(self, target):
        """
        :type target: str
        :rtype: str
        """
        x, y = 0, 0
        result = []
        for c in target:
            y1, x1 = divmod(ord(c)-ord('a'), 5)
            result.append('U' * max(y-y1, 0))
            result.append('L' * max(x-x1, 0))
            result.append('R' * max(x1-x, 0))
            result.append('D' * max(y1-y, 0))
            result.append('!')
            x, y = x1, y1
        return "".join(result)

# 23 February 2024

# Time:  O(n)
# Space: O(n)

# freq table
class Solution(object):
    def edgeScore(self, edges):
        """
        :type edges: List[int]
        :rtype: int
        """
        score = [0]*len(edges)
        for u, v in enumerate(edges):
            score[v] += u
        return max(xrange(len(edges)), key=lambda x:score[x])
    

# 24 February 2024

# Time:  O(n^3 / k)
# Space: O(n^2)

class Solution(object):
    def mergeStones(self, stones, K):
        """
        :type stones: List[int]
        :type K: int
        :rtype: int
        """
        if (len(stones)-1) % (K-1):
            return -1
        prefix = [0]
        for x in stones:
            prefix.append(prefix[-1]+x)
        dp = [[0]*len(stones) for _ in xrange(len(stones))]
        for l in xrange(K-1, len(stones)):
            for i in xrange(len(stones)-l):
                dp[i][i+l] = min(dp[i][j]+dp[j+1][i+l] for j in xrange(i, i+l, K-1))
                if l % (K-1) == 0:
                    dp[i][i+l] += prefix[i+l+1] - prefix[i]
        return dp[0][len(stones)-1]

# 25 February 2024

# Time:  O(n^2)
# Space: O(1)

# array
class Solution(object):
    def findChampion(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return next(u for u in xrange(len(grid)) if sum(grid[u]) == len(grid)-1)

# 26 February 2024

# Time:  O(n)
# Space: O(n)

import collections


class Solution(object):
    def largestUniqueNumber(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        A.append(-1)
        return max(k for k,v in collections.Counter(A).items() if v == 1)

# 27 February 2024

# Time:  O(n)
# Space: O(h)

# Definition for Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, random=None):
        self.val = val
        self.left = left
        self.right = right
        self.random = random


# Definition for NodeCopy.
class NodeCopy(object):
    def __init__(self, val=0, left=None, right=None, random=None):
        pass


class Solution(object):
    def copyRandomBinaryTree(self, root):
        """
        :type root: Node
        :rtype: NodeCopy
        """
        def iter_dfs(node, callback):
            result = None
            stk = [node]
            while stk:
                node = stk.pop()
                if not node:
                    continue
                left_node, copy = callback(node)
                if not result:
                    result = copy
                stk.append(node.right)
                stk.append(left_node)
            return result
    
        def merge(node):
            copy = NodeCopy(node.val)
            node.left, copy.left = copy, node.left
            return copy.left, copy
        
        def clone(node):
            copy = node.left
            node.left.random = node.random.left if node.random else None
            node.left.right = node.right.left if node.right else None
            return copy.left, copy
        
        def split(node):
            copy = node.left
            node.left, copy.left = copy.left, copy.left.left if copy.left else None
            return node.left, copy
    
        iter_dfs(root, merge)
        iter_dfs(root, clone)
        return iter_dfs(root, split)


# Time:  O(n)
# Space: O(h)
class Solution_Recu(object):
    def copyRandomBinaryTree(self, root):
        """
        :type root: Node
        :rtype: NodeCopy
        """
        def dfs(node, callback):
            if not node:
                return None
            left_node, copy = callback(node)
            dfs(left_node, callback)
            dfs(node.right, callback) 
            return copy
    
        def merge(node):
            copy = NodeCopy(node.val)
            node.left, copy.left = copy, node.left
            return copy.left, copy
        
        def clone(node):
            copy = node.left
            node.left.random = node.random.left if node.random else None
            node.left.right = node.right.left if node.right else None
            return copy.left, copy
        
        def split(node):
            copy = node.left
            node.left, copy.left = copy.left, copy.left.left if copy.left else None
            return node.left, copy
    
        dfs(root, merge)
        dfs(root, clone)
        return dfs(root, split)


# Time:  O(n)
# Space: O(n)
import collections


class Solution2(object):
    def copyRandomBinaryTree(self, root):
        """
        :type root: Node
        :rtype: NodeCopy
        """ 
        lookup = collections.defaultdict(lambda: NodeCopy())
        lookup[None] = None
        stk = [root]
        while stk:
            node = stk.pop()
            if not node:
                continue
            lookup[node].val = node.val
            lookup[node].left = lookup[node.left]
            lookup[node].right = lookup[node.right]
            lookup[node].random = lookup[node.random]
            stk.append(node.right)
            stk.append(node.left)
        return lookup[root]


# Time:  O(n)
# Space: O(n)
import collections


class Solution2_Recu(object):
    def copyRandomBinaryTree(self, root):
        """
        :type root: Node
        :rtype: NodeCopy
        """ 
        def dfs(node, lookup):
            if not node:
                return
            lookup[node].val = node.val
            lookup[node].left = lookup[node.left]
            lookup[node].right = lookup[node.right]
            lookup[node].random = lookup[node.random]
            dfs(node.left, lookup)
            dfs(node.right, lookup)
    
        lookup = collections.defaultdict(lambda: NodeCopy())
        lookup[None] = None
        dfs(root, lookup)
        return lookup[root]

# 28 February 2024

# Time:  O((n * log(max_num)) * logn)
# Space: O(n)

import heapq


class Solution(object):
    def minimumDeviation(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_heap = [-num*2 if num%2 else -num for num in nums]
        heapq.heapify(max_heap)
        min_elem = -max(max_heap)
        result = float("inf")
        while len(max_heap) == len(nums):
            num = -heapq.heappop(max_heap)
            result = min(result, num-min_elem)
            if not num%2:
                min_elem = min(min_elem, num//2)
                heapq.heappush(max_heap, -num//2)
        return result

# 29 February 2024

# Time:  O(n^2)
# Space: O(n)

# dp
class Solution(object):
    def minimumBeautifulSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_pow_5 = 1
        while max_pow_5*5 <= (1<<len(s))-1:
            max_pow_5 *= 5
        dp = [float("inf")]*(len(s)+1)
        dp[0] = 0
        for i in xrange(len(s)):
            if s[i] == '0':
                continue
            curr = 0
            for j in xrange(i, len(s)):
                curr = curr*2+int(s[j])
                if max_pow_5%curr == 0:
                    dp[j+1] = min(dp[j+1], dp[(i-1)+1]+1)
        return dp[-1] if dp[-1] != float("inf") else -1


# Time:  O(n^2)
# Space: O(n)
# dp
class Solution2(object):
    def minimumBeautifulSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_pow_5 = 1
        while max_pow_5*5 <= (1<<len(s))-1:
            max_pow_5 *= 5
        dp = [float("inf")]*(len(s)+1)
        dp[0] = 0
        for i in xrange(len(s)):
            curr = 0
            for j in reversed(xrange(i+1)):
                curr += int(s[j])<<(i-j)
                if s[j] == '1' and max_pow_5%curr == 0:
                    dp[i+1] = min(dp[i+1], dp[(j-1)+1]+1)
        return dp[-1] if dp[-1] != float("inf") else -1

# 01 March 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def maxDepth(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = curr = 0
        for c in s:
            if c == '(':
                curr += 1
                result = max(result, curr)
            elif c == ')':
                curr -= 1
        return result

# 02 March 2024

# Time:  O(logn), pow is O(logn).
# Space: O(1)

class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 4:
            return n - 1

        #  Proof.
        #  1. Let n = a1 + a2 + ... + ak, product = a1 * a2 * ... * ak
        #      - For each ai >= 4, we can always maximize the product by:
        #        ai <= 2 * (ai - 2)
        #      - For each aj >= 5, we can always maximize the product by:
        #        aj <= 3 * (aj - 3)
        #
        #     Conclusion 1:
        #      - For n >= 4, the max of the product must be in the form of
        #        3^a * 2^b, s.t. 3a + 2b = n
        #
        #  2. To maximize the product = 3^a * 2^b s.t. 3a + 2b = n
        #      - For each b >= 3, we can always maximize the product by:
        #        3^a * 2^b <= 3^(a+2) * 2^(b-3) s.t. 3(a+2) + 2(b-3) = n
        #
        #     Conclusion 2:
        #      - For n >= 4, the max of the product must be in the form of
        #        3^Q * 2^R, 0 <= R < 3 s.t. 3Q + 2R = n
        #        i.e.
        #          if n = 3Q + 0,   the max of the product = 3^Q * 2^0
        #          if n = 3Q + 2,   the max of the product = 3^Q * 2^1
        #          if n = 3Q + 2*2, the max of the product = 3^Q * 2^2

        res = 0
        if n % 3 == 0:            #  n = 3Q + 0, the max is 3^Q * 2^0
            res = 3 ** (n // 3)
        elif n % 3 == 2:          #  n = 3Q + 2, the max is 3^Q * 2^1
            res = 3 ** (n // 3) * 2
        else:                     #  n = 3Q + 4, the max is 3^Q * 2^2
            res = 3 ** (n // 3 - 1) * 4
        return res


# Time:  O(n)
# Space: O(1)
# DP solution.
class Solution2(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 4:
            return n - 1

        # integerBreak(n) = max(integerBreak(n - 2) * 2, integerBreak(n - 3) * 3)
        res = [0, 1, 2, 3]
        for i in xrange(4, n + 1):
            res[i % 4] = max(res[(i - 2) % 4] * 2, res[(i - 3) % 4] * 3)
        return res[n % 4]

# 03 March 2024

# Time:  O(n^p) = O(1), n is the max number of possible moves for each piece, and n is at most 29
#                     , p is the number of pieces, and p is at most 4
# Space: O(1)

class Solution(object):
    def countCombinations(self, pieces, positions):
        """
        :type pieces: List[str]
        :type positions: List[List[int]]
        :rtype: int
        """
        directions = {"rook": [(0, 1), (1, 0), (0, -1), (-1, 0)],
                      "bishop": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
                      "queen" : [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]}
        all_mask = 2**7-1  # at most 7 seconds in 8x8 board
        def backtracking(pieces, positions, i, lookup):
            if i == len(pieces):
                return 1
            result = 0
            r, c = positions[i]
            r, c = r-1, c-1
            mask = all_mask
            if not (lookup[r][c]&mask):
                lookup[r][c] += mask  # stopped at (r, c)
                result += backtracking(pieces, positions, i+1, lookup)
                lookup[r][c] -= mask          
            for dr, dc in directions[pieces[i]]:
                bit, nr, nc = 1, r+dr, c+dc
                mask = all_mask  # (mask&bit == 1): (log2(bit)+1)th second is occupied
                while 0 <= nr < 8 and 0 <= nc < 8 and not (lookup[nr][nc]&bit):
                    lookup[nr][nc] += bit
                    mask -= bit
                    if not (lookup[nr][nc]&mask):  # stopped at (nr, nc)
                        lookup[nr][nc] += mask
                        result += backtracking(pieces, positions, i+1, lookup)
                        lookup[nr][nc] -= mask
                    bit, nr, nc = bit<<1, nr+dr, nc+dc
                while bit>>1:
                    bit, nr, nc = bit>>1, nr-dr, nc-dc
                    lookup[nr][nc] -= bit
            return result

        return backtracking(pieces, positions, 0, [[0]*8 for _ in range(8)])

# 04 March 2024

# Time:  O(n)
# Space: O(1)

# simulation
class Solution(object):
    def calculateTax(self, brackets, income):
        """
        :type brackets: List[List[int]]
        :type income: int
        :rtype: float
        """
        result = prev = 0
        for u, p in brackets:
            result += max((min(u, income)-prev)*p/100.0, 0.0)
            prev = u
        return result

# 05 March 2024

# Time:  O(n * 2^n)
# Space: O(2^n)

# bitmasks, dp
class Solution(object):
    def minimumTime(self, power):
        """
        :type power: List[int]
        :rtype: int
        """
        def ceil_divide(a, b):
            return (a+b-1)//b

        INF = float("inf")
        dp = {0:0}
        for gain in xrange(1, len(power)+1):
            new_dp = collections.defaultdict(lambda:INF)
            for mask in dp.iterkeys():
                for i in xrange(len(power)):
                    if mask&(1<<i) == 0:
                        new_dp[mask|(1<<i)] = min(new_dp[mask|(1<<i)], dp[mask]+ceil_divide(power[i], gain))
            dp = new_dp
        return dp[(1<<len(power))-1]

# 06 March 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def prefixesDivBy5(self, A):
        """
        :type A: List[int]
        :rtype: List[bool]
        """
        for i in xrange(1, len(A)):
            A[i] += A[i-1] * 2 % 5
        return [x % 5 == 0 for x in A]

# 07 March 2024

# Time:  O(n)
# Space: O(logn)

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self.sortedArrayToBSTRecu(nums, 0, len(nums))

    def sortedArrayToBSTRecu(self, nums, start, end):
        if start == end:
            return None
        mid = start + self.perfect_tree_pivot(end - start)
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBSTRecu(nums, start, mid)
        node.right = self.sortedArrayToBSTRecu(nums, mid + 1, end)
        return node

    def perfect_tree_pivot(self, n):
        """
        Find the point to partition n keys for a perfect binary search tree
        """
        x = 1
        # find a power of 2 <= n//2
        # while x <= n//2:  # this loop could probably be written more elegantly :)
        #     x *= 2
        x = 1 << (n.bit_length() - 1)  # use the left bit shift, same as multiplying x by 2**n-1

        if x // 2 - 1 <= (n - x):
            return x - 1  # case 1: the left subtree of the root is perfect and the right subtree has less nodes
        else:
            return n - x // 2  # case 2 == n - (x//2 - 1) - 1 : the left subtree of the root
                               # has more nodes and the right subtree is perfect.

# Time:  O(n)
# Space: O(logn)
class Solution2(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        self.iterator = iter(nums)
        return self.helper(0, len(nums))
    
    def helper(self, start, end):
        if start == end:
            return None
        
        mid = (start + end) // 2
        left = self.helper(start, mid)
        current = TreeNode(next(self.iterator))
        current.left = left
        current.right = self.helper(mid+1, end)
        return current

# 08 March 2024

# Time:  O(n * alpha(n)) = O(n)
# Space: O(n)

class UnionFind(object):  # Time: O(n * alpha(n)), Space: O(n)
    def __init__(self, n):
        self.set = range(n)
        self.rank = [0]*n
        self.right = range(n)  # added

    def find_set(self, x):
        stk = []
        while self.set[x] != x:  # path compression
            stk.append(x)
            x = self.set[x]
        while stk:
            self.set[stk.pop()] = x
        return x

    def union_set(self, x, y):
        x, y = self.find_set(x), self.find_set(y)
        if x == y:
            return False
        if self.rank[x] > self.rank[y]:  # union by rank
            x, y = y, x
        self.set[x] = self.set[y]
        if self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        self.right[y] = max(self.right[x], self.right[y])  # added
        return True

    def right_set(self, x):  # added
        return self.right[self.find_set(x)]


# bfs, union find
class Solution(object):
    def minReverseOperations(self, n, p, banned, k):
        """
        :type n: int
        :type p: int
        :type banned: List[int]
        :type k: int
        :rtype: List[int]
        """
        lookup = [False]*n
        for i in banned:
            lookup[i] = True
        d = 0
        result = [-1]*n
        result[p] = d
        uf = UnionFind(n+2)
        uf.union_set(p, p+2)
        q = [p]
        d += 1
        while q:
            new_q = []
            for p in q:
                left, right = 2*max(p-(k-1), 0)+(k-1)-p, 2*min(p+(k-1), n-1)-(k-1)-p
                p = uf.right_set(left)
                while p <= right:
                    if not lookup[p]:
                        result[p] = d
                        new_q.append(p)
                    uf.union_set(p, p+2)
                    p = uf.right_set(p)
            q = new_q
            d += 1
        return result


# Time:  O(nlogn)
# Space: O(n)
from sortedcontainers import SortedList


# bfs, sorted list
class Solution2(object):
    def minReverseOperations(self, n, p, banned, k):
        """
        :type n: int
        :type p: int
        :type banned: List[int]
        :type k: int
        :rtype: List[int]
        """
        lookup = [False]*n
        for i in banned:
            lookup[i] = True
        d = 0
        result = [-1]*n
        result[p] = d
        sl = [SortedList(i for i in xrange(0, n, 2)), SortedList(i for i in xrange(1, n, 2))]
        sl[p%2].remove(p)
        q = [p]
        d += 1
        while q:
            new_q = []
            for p in q:
                left, right = 2*max(p-(k-1), 0)+(k-1)-p, 2*min(p+(k-1), n-1)-(k-1)-p
                for p in list(sl[left%2].irange(left, right)):
                    if not lookup[p]:
                        result[p] = d
                        new_q.append(p)
                    sl[left%2].remove(p)
            q = new_q
            d += 1
        return result

# 09 March 2024

# Time:  O(1)
# Space: O(1)

# math
class Solution(object):
    def differenceOfSums(self, n, m):
        """
        :type n: int
        :type m: int
        :rtype: int
        """
        def arithmetic_progression_sum(a, d, l):
            return (a+(a+(l-1)*d))*l//2
    
        return arithmetic_progression_sum(1, 1, n) - 2*arithmetic_progression_sum(m, m, n//m)


# Time:  O(1)
# Space: O(1)
# math
class Solution2(object):
    def differenceOfSums(self, n, m):
        """
        :type n: int
        :type m: int
        :rtype: int
        """
        return (n+1)*n//2 - 2*(((n//m+1)*(n//m)//2)*m)

# 10 March 2024

# Time:  O(u + klogk), k is most recently number of tweets,
#                      u is the number of the user's following.
# Space: O(t + f), t is the total number of tweets,
#                  f is the total number of followings.

import collections
import heapq
import random


class Twitter(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.__number_of_most_recent_tweets = 10
        self.__followings = collections.defaultdict(set)
        self.__messages = collections.defaultdict(list)
        self.__time = 0

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        self.__time += 1
        self.__messages[userId].append((self.__time, tweetId))

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        def nth_element(nums, n, compare=lambda a, b: a < b):
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

            left, right = 0, len(nums)-1
            while left <= right:
                pivot_idx = random.randint(left, right)
                pivot_left, pivot_right = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left-1
                else:  # pivot_right < n.
                    left = pivot_right+1

        candidates = []
        if self.__messages[userId]:
            candidates.append((-self.__messages[userId][-1][0], userId, 0))
        for uid in self.__followings[userId]:
            if self.__messages[uid]:
                candidates.append((-self.__messages[uid][-1][0], uid, 0))
        nth_element(candidates, self.__number_of_most_recent_tweets-1)
        max_heap = candidates[:self.__number_of_most_recent_tweets]
        heapq.heapify(max_heap)
        result = []
        while max_heap and len(result) < self.__number_of_most_recent_tweets:
            t, uid, curr = heapq.heappop(max_heap)
            nxt = curr + 1
            if nxt != len(self.__messages[uid]):
                heapq.heappush(max_heap, (-self.__messages[uid][-(nxt+1)][0], uid, nxt))
            result.append(self.__messages[uid][-(curr+1)][1])
        return result

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId != followeeId:
            self.__followings[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.__followings[followerId].discard(followeeId)

# 11 March 2024

# Time:  O(n)
# Space: O(1)

# greedy, math
class Solution(object):
    def minimumReplacement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def ceil_divide(a, b):
            return (a+b-1)//b

        result = 0
        curr = nums[-1]
        for x in reversed(nums):
            cnt = ceil_divide(x, curr)
            result += cnt-1
            curr = x//cnt
        return result

# 12 March 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def makeGood(self, s):
        """
        :type s: str
        :rtype: str
        """
        stk = []
        for ch in s:
            counter_ch = ch.upper() if ch.islower() else ch.lower()
            if stk and stk[-1] == counter_ch:
                stk.pop()
            else:
                stk.append(ch)
        return "".join(stk)

# 13 March 2024

# Time:  O(n^2) ~ O(n^3)
# Space: O(n^2)

import collections
import itertools


class Solution(object):
    def minAreaFreeRect(self, points):
        """
        :type points: List[List[int]]
        :rtype: float
        """
        points.sort()
        points = [complex(*z) for z in points]
        lookup = collections.defaultdict(list)
        for P, Q in itertools.combinations(points, 2):
            lookup[P-Q].append((P+Q) / 2)

        result = float("inf")
        for A, candidates in lookup.iteritems():
            for P, Q in itertools.combinations(candidates, 2):
                if A.real * (P-Q).real + A.imag * (P-Q).imag == 0.0:
                    result = min(result, abs(A) * abs(P-Q))
        return result if result < float("inf") else 0.0

# 14 March 2024

# Time:  O(1)
# Space: O(1)

class Solution(object):
    def complexNumberMultiply(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        ra, ia = map(int, a[:-1].split('+'))
        rb, ib = map(int, b[:-1].split('+'))
        return '%d+%di' % (ra * rb - ia * ib, ra * ib + ia * rb)

# 15 March 2024

# Time:  O((m * n)^2 * (m + n))
# Space: O((m * n)^2)

import collections


class Solution(object):
    def canMouseWin(self, grid, catJump, mouseJump):
        """
        :type grid: List[str]
        :type catJump: int
        :type mouseJump: int
        :rtype: bool
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        DRAW, MOUSE, CAT = range(3)
        def parents(m, c, t):
            if t == CAT:
                for nm in graph[m, MOUSE^CAT^t]:
                    yield nm, c, MOUSE^CAT^t
            else:
                for nc in graph[c, MOUSE^CAT^t]:
                    yield m, nc, MOUSE^CAT^t

        R, C = len(grid), len(grid[0])
        N = R*C
        WALLS = set()
        FOOD, MOUSE_START, CAT_START = [-1]*3
        for r in xrange(R):
            for c in xrange(C):
                if grid[r][c] == 'M':
                    MOUSE_START = r*C + c
                elif grid[r][c] == 'C':
                    CAT_START = r*C + c
                elif grid[r][c] == 'F':
                    FOOD = r*C + c
                elif grid[r][c] == '#':
                    WALLS.add(r*C + c)

        graph = collections.defaultdict(set)
        jump = {MOUSE:mouseJump, CAT:catJump}
        for r in xrange(R):
            for c in xrange(C):
                if grid[r][c] == '#':
                    continue
                pos = r*C + c
                for t in [MOUSE, CAT]:
                    for dr, dc in directions:
                        for d in xrange(jump[t]+1):
                            nr, nc = r+dr*d, c+dc*d
                            if not (0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#'):
                                break
                            graph[pos, t].add(nr*C + nc)

        degree = {}
        for m in xrange(N):
            for c in xrange(N):
                degree[m, c, MOUSE] = len(graph[m, MOUSE])
                degree[m, c, CAT] = len(graph[c, CAT])
        color = collections.defaultdict(int)
        q = collections.deque()
        for i in xrange(N):
            if i in WALLS or i == FOOD:
                continue
            color[FOOD, i, CAT] = MOUSE
            q.append((FOOD, i, CAT, MOUSE))
            color[i, FOOD, MOUSE] = CAT
            q.append((i, FOOD, MOUSE, CAT))
            for t in [MOUSE, CAT]:
                color[i, i, t] = CAT
                q.append((i, i, t, CAT))
        while q:
            i, j, t, c = q.popleft()
            for ni, nj, nt in parents(i, j, t):
                if color[ni, nj, nt] != DRAW:
                    continue
                if nt == c:
                    color[ni, nj, nt] = c
                    q.append((ni, nj, nt, c))
                    continue
                degree[ni, nj, nt] -= 1
                if not degree[ni, nj, nt]:
                    color[ni, nj, nt] = c
                    q.append((ni, nj, nt, c))
        return color[MOUSE_START, CAT_START, MOUSE] == MOUSE


# Time:  O((m * n)^2 * (m + n))
# Space: O((m * n)^2)
import collections


class Solution2(object):
    def canMouseWin(self, grid, catJump, mouseJump):
        """
        :type grid: List[str]
        :type catJump: int
        :type mouseJump: int
        :rtype: bool
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        DRAW, MOUSE, CAT = range(3)
        def parents(m, c, t):
            if t == CAT:
                for nm in graph[m, MOUSE^CAT^t]:
                    yield nm, c, MOUSE^CAT^t
            else:
                for nc in graph[c, MOUSE^CAT^t]:
                    yield m, nc, MOUSE^CAT^t

        R, C = len(grid), len(grid[0])
        N = R*C
        WALLS = set()
        FOOD, MOUSE_START, CAT_START = [-1]*3
        for r in xrange(R):
            for c in xrange(C):
                if grid[r][c] == 'M':
                    MOUSE_START = r*C + c
                elif grid[r][c] == 'C':
                    CAT_START = r*C + c
                elif grid[r][c] == 'F':
                    FOOD = r*C + c
                elif grid[r][c] == '#':
                    WALLS.add(r*C + c)
        graph = collections.defaultdict(set)
        jump = {MOUSE:mouseJump, CAT:catJump}
        for r in xrange(R):
            for c in xrange(C):
                if grid[r][c] == '#':
                    continue
                pos = r*C + c
                for t in [MOUSE, CAT]:
                    for dr, dc in directions:
                        for d in xrange(jump[t]+1):
                            nr, nc = r+dr*d, c+dc*d
                            if not (0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#'):
                                break
                            graph[pos, t].add(nr*C + nc)

        degree = {}
        for m in xrange(N):
            for c in xrange(N):
                # degree[m, c, MOUSE] = len(graph[m, MOUSE])
                degree[m, c, CAT] = len(graph[c, CAT])
        color = collections.defaultdict(int)
        q1 = collections.deque()
        # q2 = collections.deque()
        for i in xrange(N):
            if i in WALLS or i == FOOD:
                continue
            color[FOOD, i, CAT] = MOUSE
            q1.append((FOOD, i, CAT))
            color[i, FOOD, MOUSE] = CAT
            # q2.append((i, FOOD, MOUSE))
            for t in [MOUSE, CAT]:
                color[i, i, t] = CAT
                # q2.append((i, i, t))
        while q1:
            i, j, t = q1.popleft()
            for ni, nj, nt in parents(i, j, t):
                if color[ni, nj, nt] != DRAW:
                    continue
                if t == CAT:
                    color[ni, nj, nt] = MOUSE
                    q1.append((ni, nj, nt))
                    continue
                degree[ni, nj, nt] -= 1
                if not degree[ni, nj, nt]:
                    color[ni, nj, nt] = MOUSE
                    q1.append((ni, nj, nt))
        # while q2:
        #     i, j, t = q2.popleft()
        #     for ni, nj, nt in parents(i, j, t):
        #         if color[ni, nj, nt] != DRAW:
        #             continue
        #         if t == MOUSE:
        #             color[ni, nj, nt] = CAT
        #             q2.append((ni, nj, nt))
        #             continue
        #         degree[ni, nj, nt] -= 1
        #         if not degree[ni, nj, nt]:
        #             color[ni, nj, nt] = CAT
        #             q2.append((ni, nj, nt))
        return color[MOUSE_START, CAT_START, MOUSE] == MOUSE

# 16 March 2024

# Time:  O(n + m)
# Space: O(1)

import collections


# freq table
class Solution(object):
    def rearrangeCharacters(self, s, target):
        """
        :type s: str
        :type target: str
        :rtype: int
        """
        cnt1 = collections.Counter(s)
        cnt2 = collections.Counter(target)
        return min(cnt1[k]//v for k, v in cnt2.iteritems())

# 17 March 2024

# Time:  O(nlogn)
# Space: O(1)

class Solution(object):
    def bagOfTokensScore(self, tokens, P):
        """
        :type tokens: List[int]
        :type P: int
        :rtype: int
        """
        tokens.sort()
        result, points = 0, 0
        left, right = 0, len(tokens)-1
        while left <= right:
            if P >= tokens[left]:
                P -= tokens[left]
                left += 1
                points += 1
                result = max(result, points)
            elif points > 0:
                points -= 1
                P += tokens[right]
                right -= 1
            else:
                break
        return result

# 18 March 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def minimumLength(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, right = 0, len(s)-1
        while left < right:
            if s[left] != s[right]:
                break
            c = s[left]
            while left <= right:
                if s[left] != c:
                    break
                left += 1
            while left <= right:
                if s[right] != c:
                    break
                right -= 1
        return right-left+1

# 19 March 2024

# Time:  O(logn)
# Space: O(1)

class Solution(object):
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left) / 2
            if not (mid%2 == 0 and mid+1 < len(nums) and \
                    nums[mid] == nums[mid+1]) and \
               not (mid%2 == 1 and nums[mid] == nums[mid-1]):
                right = mid-1
            else:
                left = mid+1
        return nums[left]

# 20 March 2024

# Time:  O(n * C(n - 1, c - 1)), n is length of str, c is unique count of pattern,
#                                there are H(n - c, c - 1) = C(n - 1, c - 1) possible splits of string,
#                                and each one costs O(n) to check if it matches the word pattern.
# Space: O(n + c)

class Solution(object):
    def wordPatternMatch(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        w2p, p2w = {}, {}
        return self.match(pattern, str, 0, 0, w2p, p2w)


    def match(self, pattern, str, i, j, w2p, p2w):
        is_match = False
        if i == len(pattern) and j == len(str):
            is_match = True
        elif i < len(pattern) and j < len(str):
            p = pattern[i]
            if p in p2w:
                w = p2w[p]
                if w == str[j:j+len(w)]:  # Match pattern.
                    is_match = self.match(pattern, str, i + 1, j + len(w), w2p, p2w)
                # Else return false.
            else:
                for k in xrange(j, len(str)):  # Try any possible word
                    w = str[j:k+1]
                    if w not in w2p:
                        # Build mapping. Space: O(n + c)
                        w2p[w], p2w[p] = p, w
                        is_match = self.match(pattern, str, i + 1, k + 1, w2p, p2w)
                        w2p.pop(w), p2w.pop(p)
                    if is_match:
                        break
        return is_match

# 21 March 2024

# Time:  O(n)
# Space: O(1)

import fractions


class Solution(object):
    def findGCD(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return fractions.gcd(min(nums), max(nums))

# 22 March 2024

# Time:  O(m + n)
# Space: O(m + n)

# hash table
class Solution(object):
    def minNumber(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        common = set(nums1)&set(nums2)
        if common:
            return min(common)
        mn1, mn2 = min(nums1), min(nums2)
        if mn1 > mn2:
            mn1, mn2 = mn2, mn1
        return 10*mn1+mn2

# 23 March 2024

# Time:  O(nlogr), r = max(nums)
# Space: O(logr)

# dp
class Solution(object):
    def subarrayGCD(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def gcd(a, b):
            while b:
                a, b = b, a%b
            return a

        result = 0
        dp = collections.Counter()
        for x in nums:
            new_dp = collections.Counter()
            if x%k == 0:
                dp[x] += 1
                for g, cnt in dp.iteritems():
                    new_dp[gcd(g, x)] += cnt
            dp = new_dp
            result += dp[k]
        return result


# Time:  O(n^2)
# Space: O(1)
# brute force
class Solution2(object):
    def subarrayGCD(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def gcd(a, b):
            while b:
                a, b = b, a%b
            return a

        result = 0
        for i in xrange(len(nums)):
            g = 0
            for j in xrange(i, len(nums)):
                if nums[j]%k:
                    break
                g = gcd(g, nums[j])
                result += int(g == k)
        return result

# 25 March 2024

# Time:  O(n * k)
# Space: O(n + k)

import heapq


# Heap solution. (620ms)
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        heap, uglies, idx, ugly_by_last_prime = [], [0] * n, [0] * len(primes), [0] * n
        uglies[0] = 1

        for k, p in enumerate(primes):
            heapq.heappush(heap, (p, k))

        for i in xrange(1, n):
            uglies[i], k = heapq.heappop(heap)
            ugly_by_last_prime[i] = k
            idx[k] += 1
            while ugly_by_last_prime[idx[k]] > k:
                idx[k] += 1
            heapq.heappush(heap, (primes[k] * uglies[idx[k]], k))

        return uglies[-1]

# Time:  O(n * k)
# Space: O(n + k)
# Hash solution. (932ms)
class Solution2(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        uglies, idx, heap, ugly_set = [0] * n, [0] * len(primes), [], set([1])
        uglies[0] = 1

        for k, p in enumerate(primes):
            heapq.heappush(heap, (p, k))
            ugly_set.add(p)

        for i in xrange(1, n):
            uglies[i], k = heapq.heappop(heap)
            while (primes[k] * uglies[idx[k]]) in ugly_set:
                idx[k] += 1
            heapq.heappush(heap, (primes[k] * uglies[idx[k]], k))
            ugly_set.add(primes[k] * uglies[idx[k]])

        return uglies[-1]

# Time:  O(n * logk) ~ O(n * klogk)
# Space: O(n + k)
class Solution3(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        uglies, idx, heap = [1], [0] * len(primes), []
        for k, p in enumerate(primes):
            heapq.heappush(heap, (p, k))

        for i in xrange(1, n):
            min_val, k = heap[0]
            uglies += [min_val]

            while heap[0][0] == min_val:  # worst time: O(klogk)
                min_val, k = heapq.heappop(heap)
                idx[k] += 1
                heapq.heappush(heap, (primes[k] * uglies[idx[k]], k))

        return uglies[-1]

# Time:  O(n * k)
# Space: O(n + k)
# TLE due to the last test case, but it passess and performs the best in C++.
class Solution4(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        uglies = [0] * n
        uglies[0] = 1
        ugly_by_prime = list(primes)
        idx = [0] * len(primes)

        for i in xrange(1, n):
            uglies[i] = min(ugly_by_prime)
            for k in xrange(len(primes)):
                if uglies[i] == ugly_by_prime[k]:
                    idx[k] += 1
                    ugly_by_prime[k] = primes[k] * uglies[idx[k]]

        return uglies[-1]

# Time:  O(n * logk) ~ O(n * klogk)
# Space: O(k^2)
# TLE due to the last test case, but it passess and performs well in C++.
class Solution5(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        ugly_number = 0

        heap = []
        heapq.heappush(heap, 1)
        for p in primes:
            heapq.heappush(heap, p)
        for _ in xrange(n):
            ugly_number = heapq.heappop(heap)
            for i in xrange(len(primes)):
                if ugly_number % primes[i] == 0:
                    for j in xrange(i + 1):
                        heapq.heappush(heap, ugly_number * primes[j])
                    break

        return ugly_number

# 26 March 2024

# Time:  O(klogn), k = len(set(nums))
# Space: O(1)

# Definition for BigArray.
class BigArray:
    def at(self, index):
        pass
    def size(self):
        pass


# binary search
class Solution(object):
    def countBlocks(self, nums):
        """
        :type nums: BigArray
        :rtype: int
        """
        def binary_search_right(left, right, check):
            while left <= right:
                mid = left + (right-left)//2
                if not check(mid):
                    right = mid-1
                else:
                    left = mid+1
            return right

        n = nums.size()
        result = left = 0
        while left != n:
            target = nums.at(left)
            left = binary_search_right(left, n-1, lambda x: nums.at(x) == target)+1
            result += 1
        return result

# 27 March 2024

# Time:  O(m * n * k)
# Space: O(n * k)

# dp
class Solution(object):
    def numberOfPaths(self, grid, k):
        """
        :type grid: List[List[int]]
        :type k: int
        :rtype: int
        """
        MOD = 10**9+7
        dp = [[0 for _ in xrange(k)] for _ in xrange(len(grid[0]))]
        dp[0][0] = 1
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                dp[j] = [((dp[j-1][(l-grid[i][j])%k] if j-1 >= 0 else 0)+dp[j][(l-grid[i][j])%k])%MOD for l in xrange(k)]
        return dp[-1][0]

# 28 March 2024

# Time:  O(1)
# Space: O(1)

class Solution(object):
    def checkOverlap(self, radius, x_center, y_center, x1, y1, x2, y2):
        """
        :type radius: int
        :type x_center: int
        :type y_center: int
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
        :rtype: bool
        """
        x1 -= x_center
        y1 -= y_center
        x2 -= x_center
        y2 -= y_center
        x = x1 if x1 > 0 else x2 if x2 < 0 else 0
        y = y1 if y1 > 0 else y2 if y2 < 0 else 0
        return x**2 + y**2 <= radius**2


# Time:  O(1)
# Space: O(1)
class Solution2(object):
    def checkOverlap(self, radius, x_center, y_center, x1, y1, x2, y2):
        """
        :type radius: int
        :type x_center: int
        :type y_center: int
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
        :rtype: bool
        """
        x1 -= x_center
        y1 -= y_center
        x2 -= x_center
        y2 -= y_center        
        x = min(abs(x1), abs(x2)) if x1*x2 > 0 else 0
        y = min(abs(y1), abs(y2)) if y1*y2 > 0 else 0
        return x**2 + y**2 <= radius**2

# 29 March 2024

# Time:  O(1)
# Space: O(1)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        pass


# tree
class Solution(object):
    def checkTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        return root.val == root.left.val+root.right.val

# 30 March 2024

# Time:  O(n)
# Space: O(1)

# math
class Solution(object):
    def averageValue(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        total = cnt = 0
        for x in nums:
            if x%6:
                continue
            total += x
            cnt += 1
        return total//cnt if cnt else 0

# 31 March 2024

# Time:  O(n)
# Space: O(1)

# dp
class Solution(object):
    def minimizeConcatenatedLength(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        dp = [[float("-inf")]*26 for _ in xrange(2)]
        dp[0][ord(words[0][-1])-ord('a')] = dp[1][ord(words[0][0])-ord('a')] = 0
        for i in xrange(1, len(words)):
            new_dp = [[float("-inf")]*26 for _ in xrange(2)]
            for right in xrange(2):
                for c in xrange(26):
                    if dp[right][c] == float("-inf"):
                        continue
                    l = c if right else ord(words[i-1][0])-ord('a')
                    r = c if not right else ord(words[i-1][-1])-ord('a')
                    new_dp[0][r] = max(new_dp[0][r], dp[right][c]+int(ord(words[i][-1])-ord('a') == l))
                    new_dp[1][l] = max(new_dp[1][l], dp[right][c]+int(r == ord(words[i][0])-ord('a')))
            dp = new_dp
        return sum(len(w) for w in words)-max(dp[right][c] for right in xrange(2) for c in xrange(26))

# 01 April 2024

# Time:  O(n^2)
# Space: O(n)

import collections


class Solution(object):
    def numTilePossibilities(self, tiles):
        """
        :type tiles: str
        :rtype: int
        """
        fact = [0.0]*(len(tiles)+1)
        fact[0] = 1.0;
        for i in xrange(1, len(tiles)+1):
            fact[i] = fact[i-1]*i
        count = collections.Counter(tiles)

        # 1. we can represent each alphabet 1..26 as generating functions:
        #    G1(x) = 1 + x^1/1! + x^2/2! + x^3/3! + ... + x^count1/count1!
        #    G2(x) = 1 + x^1/1! + x^2/2! + x^3/3! + ... + x^count2/count2!
        #    ...
        #    G26(x) = 1 + x^1/1! + x^2/2! + x^3/3! + ... + x^count26/count26!
        #
        # 2. let G1(x)*G2(x)*...*G26(x) = c0 + c1*x1 + ... + ck*x^k, k is the max number s.t. ck != 0
        #    => ci (1 <= i <= k) is the number we need to divide when permuting i letters
        #    => the answer will be : c1*1! + c2*2! + ... + ck*k!
        
        coeff = [0.0]*(len(tiles)+1)
        coeff[0] = 1.0
        for i in count.itervalues():
            new_coeff = [0.0]*(len(tiles)+1)
            for j in xrange(len(coeff)):
                for k in xrange(i+1):
                    if k+j >= len(new_coeff):
                        break
                    new_coeff[j+k] += coeff[j]*1.0/fact[k]
            coeff = new_coeff

        result = 0
        for i in xrange(1, len(coeff)):
            result += int(round(coeff[i]*fact[i]))
        return result


# Time:  O(r), r is the value of result
# Space: O(n)
class Solution2(object):
    def numTilePossibilities(self, tiles):
        """
        :type tiles: str
        :rtype: int
        """
        def backtracking(counter):
            total = 0
            for k, v in counter.iteritems():
                if not v:
                    continue
                counter[k] -= 1
                total += 1+backtracking(counter)
                counter[k] += 1
            return total

        return backtracking(collections.Counter(tiles))

# 02 April 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    # @param {TreeNode} root
    # @param {TreeNode} p
    # @param {TreeNode} q
    # @return {TreeNode}
    def lowestCommonAncestor(self, root, p, q):
        s, b = sorted([p.val, q.val])
        while not s <= root.val <= b:
            # Keep searching since root is outside of [s, b].
            root = root.left if s <= root.val else root.right
        # s <= root.val <= b.
        return root

# 03 April 2024

# Time:  O(m * n) on average
# Space: O(m * n)

import random


class Solution(object):
    def kthLargestValue(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        def nth_element(nums, n, compare=lambda a, b: a < b):
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

            left, right = 0, len(nums)-1
            while left <= right:
                pivot_idx = random.randint(left, right)
                pivot_left, pivot_right = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left-1
                else:  # pivot_right < n.
                    left = pivot_right+1
        
        
        vals = []
        for r in xrange(len(matrix)):
            curr = 0
            for c in xrange(len(matrix[0])):
                curr = curr^matrix[r][c]
                if r == 0:
                    matrix[r][c] = curr
                else:
                    matrix[r][c] = curr^matrix[r-1][c]
                vals.append(matrix[r][c])
        nth_element(vals, k-1, compare=lambda a, b: a > b)
        return vals[k-1]

# 04 April 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def movesToMakeZigzag(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = [0, 0]
        for i in xrange(len(nums)):
            left = nums[i-1] if i-1 >= 0 else float("inf")
            right = nums[i+1] if i+1 < len(nums) else float("inf")
            result[i%2] += max(nums[i] - min(left, right) + 1, 0)
        return min(result)

# 05 April 2024

# Time:  O(n)
# Space: O(1)

import heapq


class Solution(object):
    def longestDiverseString(self, a, b, c):
        """
        :type a: int
        :type b: int
        :type c: int
        :rtype: str
        """
        max_heap = []
        if a:
            heapq.heappush(max_heap, (-a, 'a'))
        if b:
            heapq.heappush(max_heap, (-b, 'b'))
        if c:
            heapq.heappush(max_heap, (-c, 'c'))
        result = []
        while max_heap:
            count1, c1 = heapq.heappop(max_heap)
            if len(result) >= 2 and result[-1] == result[-2] == c1:
                if not max_heap:
                    return "".join(result)
                count2, c2 = heapq.heappop(max_heap)
                result.append(c2)
                count2 += 1
                if count2:
                    heapq.heappush(max_heap, (count2, c2))
                heapq.heappush(max_heap, (count1, c1))
                continue
            result.append(c1)
            count1 += 1
            if count1 != 0:
                heapq.heappush(max_heap, (count1, c1))
        return "".join(result)


# Time:  O(n)
# Space: O(1)
class Solution2(object):
    def longestDiverseString(self, a, b, c):
        """
        :type a: int
        :type b: int
        :type c: int
        :rtype: str
        """
        choices = [[a, 'a'], [b, 'b'], [c, 'c']]
        result = []
        for _ in xrange(a+b+c):
            choices.sort(reverse=True)
            for i, (x, c) in enumerate(choices):
                if x and result[-2:] != [c, c]:
                    result.append(c)
                    choices[i][0] -= 1
                    break
            else:
                break
        return "".join(result)

# 06 April 2024

# Time:  O(n + rlogr), r is the number of messages
# Space: O(1)

# brute force, linear search (binary search doesn't work)
class Solution(object):
    def splitMessage(self, message, limit):
        """
        :type message: str
        :type limit: int
        :rtype: List[str]
        """
        cnt, l, total, base = 1, 1, len(message)+1, 1
        while 3+l*2 < limit:
            if total+(3+l)*cnt <= limit*cnt:
                break
            cnt += 1
            if cnt == base*10:
                l += 1
                base *= 10
            total += l
        if 3+l*2 >= limit:
            return []
        result = []
        j = 0
        for i in xrange(cnt):
            l = limit-(3+len(str(i+1))+len(str(cnt)))
            result.append("%s<%s/%s>"%(message[j:j+l], i+1, cnt))
            j += l
        return result

# 07 April 2024

# Time:  O(n)
# Space: O(n)

import itertools


class Solution(object):
    def flipgame(self, fronts, backs):
        """
        :type fronts: List[int]
        :type backs: List[int]
        :rtype: int
        """
        same = {n for i, n in enumerate(fronts) if n == backs[i]}
        result = float("inf")
        for n in itertools.chain(fronts, backs):
            if n not in same:
                result = min(result, n)
        return result if result < float("inf") else 0

# 08 April 2024

# Time:  O(logn)
# Space: O(1)

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while left <= right:
            mid = left + (right - left) / 2
            if isBadVersion(mid): # noqa
                right = mid - 1
            else:
                left = mid + 1
        return left

# 09 April 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        result = []
        for x in asteroids:
            if x > 0:
                result.append(x)
                continue
            while result and 0 < result[-1] < -x:
                result.pop()
            if result and 0 < result[-1]:
                if result[-1] == -x:
                    result.pop()
                continue
            result.append(x)
        return result


# Time:  O(n)
# Space: O(n)
class Solution2(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        result = []
        for x in asteroids:
            while result and x < 0 < result[-1]:
                if result[-1] < -x:
                    result.pop()
                    continue
                elif result[-1] == -x:
                    result.pop()
                break
            else:
                result.append(x)
        return result

# 10 April 2024

# Time:  O(n)
# Space: O(n)

class UnionFind(object):
    def __init__(self, n):
        self.set = range(n)

    def find_set(self, x):
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])  # path compression.
        return self.set[x]

    def union_set(self, x, y):
        x_root, y_root = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[min(x_root, y_root)] = max(x_root, y_root)
        return True


class Solution(object):
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """
        MAX_ROW = 10000
        union_find = UnionFind(2*MAX_ROW)
        for r, c in stones:
            union_find.union_set(r, c+MAX_ROW)
        return len(stones) - len({union_find.find_set(r) for r, _ in stones})

# 11 April 2024

# Time:  O(n)
# Space: O(1)

# prefix sum
class Solution(object):
    def maxScoreIndices(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result = []
        mx = zeros = 0
        total = sum(nums)
        for i in xrange(len(nums)+1):
            zeros += ((nums[i-1] if i else 0) == 0)
            if zeros+(total-(i-zeros)) > mx:
                mx = zeros+(total-(i-zeros))
                result = []
            if zeros+(total-(i-zeros)) == mx:
                result.append(i)
        return result

# 12 April 2024

# Time:  O(n)
# Space: O(n)

# hash table
class Solution(object):
    def findMaxK(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        lookup = set(nums)
        return max([x for x in lookup if x > 0 and -x in lookup] or [-1])

# 13 April 2024

# Time:  O(n!)
# Space: O(n)


class Solution(object):
    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        def countArrangementHelper(n, arr):
            if n <= 0:
                return 1
            count = 0
            for i in xrange(n):
                if arr[i] % n == 0 or n % arr[i] == 0:
                    arr[i], arr[n-1] = arr[n-1], arr[i]
                    count += countArrangementHelper(n - 1, arr)
                    arr[i], arr[n-1] = arr[n-1], arr[i]
            return count

        return countArrangementHelper(N, range(1, N+1))

# 14 April 2024

# Time:  O(nlogn)
# Space: O(n)

import itertools
import heapq


class Solution(object):
    def maxPerformance(self, n, speed, efficiency, k):
        """
        :type n: int
        :type speed: List[int]
        :type efficiency: List[int]
        :type k: int
        :rtype: int
        """
        MOD = 10**9 + 7
        result, s_sum = 0, 0
        min_heap = []
        for e, s in sorted(itertools.izip(efficiency, speed), reverse=True):
            s_sum += s
            heapq.heappush(min_heap, s)
            if len(min_heap) > k:
                s_sum -= heapq.heappop(min_heap)
            result = max(result, s_sum*e)
        return result % MOD

# 15 April 2024

# Time:  O(n)
# Space: O(1)

class Codec(object):

    def encode(self, strs):
        """Encodes a list of strings to a single string.

        :type strs: List[str]
        :rtype: str
        """
        encoded_str = ""
        for s in strs:
            encoded_str += "%0*x" % (8, len(s)) + s
        return encoded_str


    def decode(self, s):
        """Decodes a single string to a list of strings.

        :type s: str
        :rtype: List[str]
        """
        i = 0
        strs = []
        while i < len(s):
            l = int(s[i:i+8], 16)
            strs.append(s[i+8:i+8+l])
            i += 8+l
        return strs

# 16 April 2024

# Time:  O(n + m), m is the max number of nums
# Space: O(m)

import collections


class Solution(object):
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        count = collections.Counter(nums)
        for i in xrange(max(nums)+1):
            count[i] += count[i-1]
        return [count[i-1] for i in nums]


# Time:  O(nlogn)
# Space: O(n)
import bisect


class Solution2(object):
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        sorted_nums = sorted(nums)
        return [bisect.bisect_left(sorted_nums, i) for i in nums]

# 17 April 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        result, left = 0, 0
        lookup = {}
        for right in xrange(len(s)):
            if s[right] in lookup:
                left = max(left, lookup[s[right]]+1)
            lookup[s[right]] = right
            result = max(result, right-left+1)
        return result

# 18 April 2024

# Time:  O(n)
# Space: O(n)

import collections
import operator


# combinatorics, dp
class Solution(object):
    def beautifulSubsets(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def count(x):
            y = x
            while y-k in cnt:
                y -= k
            dp = [1, 0]  # dp[0]: count without i, dp[1]: count with i
            for i in xrange(y, x+1, k):
                dp = [dp[0]+dp[1], dp[0]*((1<<cnt[i])-1)]
            return sum(dp)

        cnt = collections.Counter(nums)
        return reduce(operator.mul, (count(i) for i in cnt.iterkeys() if i+k not in cnt))-1

# 19 April 2024

# Time:  O(1)
# Space: O(1)

class Solution(object):
    def divisorGame(self, N):
        """
        :type N: int
        :rtype: bool
        """
        # 1. if we get an even, we can choose x = 1
        #    to make the opponent always get an odd
        # 2. if the opponent gets an odd, he can only choose x = 1 or other odds
        #    and we can still get an even
        # 3. at the end, the opponent can only choose x = 1 and we win
        # 4. in summary, we win if only if we get an even and 
        #    keeps even until the opponent loses
        return N % 2 == 0


# Time:  O(n^3/2)
# Space: O(n)
# dp solution
class Solution2(object):
    def divisorGame(self, N):
        """
        :type N: int
        :rtype: bool
        """
        def memoization(N, dp):
            if N == 1:
                return False
            if N not in dp:
                result = False
                for i in xrange(1, N+1):
                    if i*i > N:
                        break
                    if N % i == 0:
                        if not memoization(N-i, dp):
                            result = True
                            break
                dp[N] = result
            return dp[N]
        
        return memoization(N, {})

# 20 April 2024

# Time:  O(nlogn)
# Space: O(1)

# sort, two pointers, sliding window
class Solution(object):
    def maximumBeauty(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort()
        left = 0
        for right in xrange(len(nums)):
            if nums[right]-nums[left] > k*2:
                left += 1
        return right-left+1

# 21 April 2024

# Time:  O(n)
# Space: O(1)

# constructive algorithms, greedy, two pointers
class Solution(object):
    def minimumOperations(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = 0
        left, right = 0, len(nums)-1
        l, r = nums[left], nums[right]
        while left < right:
            if l == r:
                left += 1
                right -= 1
                l, r = nums[left], nums[right]
                continue
            if l < r:
                left += 1
                l += nums[left]
            else:
                right -= 1
                r += nums[right]
            result += 1
        return result
            

# 22 April 2024

# Time:  O(n)
# Space: O(1)

from operator import xor
from functools import reduce


class Solution(object):
    def xorGame(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return reduce(xor, nums) == 0 or \
            len(nums) % 2 == 0

# 23 April 2024

# Time:  O(n)
# Space: O(n)

import collections


class Solution(object):
    def maxOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count = collections.Counter()
        result = 0
        for x in nums:
            if k-x in count and count[k-x]:
                count[k-x] -= 1
                result += 1
            else:
                count[x] += 1
        return result

# 24 April 2024

# Time:  O(nlogn)
# Space: O(n)

import collections
from sortedcontainers import SortedList


# bit, fenwick tree, sorted list, math
class Solution(object):
    def sumCounts(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        MOD = 10**9+7
        class BIT(object):  # 0-indexed.
            def __init__(self, n):
                self.__bit = [0]*(n+1)  # Extra one for dummy node.

            def add(self, i, val):
                i += 1  # Extra one for dummy node.
                while i < len(self.__bit):
                    self.__bit[i] = (self.__bit[i]+val) % MOD
                    i += (i & -i)

            def query(self, i):
                i += 1  # Extra one for dummy node.
                ret = 0
                while i > 0:
                    ret = (ret+self.__bit[i]) % MOD
                    i -= (i & -i)
                return ret

        def update(accu, d):
            i = sl.bisect_left(idxs[x][-1])
            accu = (accu + d*(len(nums)*(2*len(sl)-1) - (2*i+1)*idxs[x][-1] - 2*(bit.query(len(nums)-1)-bit.query(idxs[x][-1])))) % MOD
            bit.add(idxs[x][-1], d*idxs[x][-1])
            return accu

        idxs = collections.defaultdict(list)
        for i in reversed(xrange(len(nums))):
            idxs[nums[i]].append(i)
        result = 0
        sl = SortedList(idxs[x][-1] for x in idxs)
        accu = (len(nums)*len(sl)**2) % MOD
        for i, x in enumerate(sl):
            accu = (accu-(2*i+1)*x) % MOD
        bit = BIT(len(nums))
        for x in sl:
            bit.add(x, x)
        for x in nums:
            result = (result+accu) % MOD  # accu = sum(count(i, k) for k in range(i, len(nums)))
            accu = update(accu, -1)
            del sl[0]
            idxs[x].pop()
            if not idxs[x]:
                continue
            sl.add(idxs[x][-1])
            accu = update(accu, +1)
        assert(accu == 0)
        return result


# Time:  O(nlogn)
# Space: O(n)
# dp, segment tree, math
class Solution2(object):
    def sumCounts(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        MOD = 10**9+7
        # Template:
        # https://github.com/kamyu104/LeetCode-Solutions/blob/master/Python/longest-substring-of-one-repeating-character.py
        class SegmentTree(object):
            def __init__(self, N,
                         build_fn=None,
                         query_fn=lambda x, y: y if x is None else x if y is None else (x+y)%MOD,
                         update_fn=lambda x, y: y if x is None else (x+y)%MOD):
                self.tree = [None]*(1<<((N-1).bit_length()+1))
                self.base = len(self.tree)>>1
                self.lazy = [None]*self.base
                self.query_fn = query_fn
                self.update_fn = update_fn
                if build_fn is not None:
                    for i in xrange(self.base, self.base+N):
                        self.tree[i] = build_fn(i-self.base)
                    for i in reversed(xrange(1, self.base)):
                        self.tree[i] = query_fn(self.tree[i<<1], self.tree[(i<<1)+1])
                self.count = [1]*len(self.tree)  # added
                for i in reversed(xrange(1, self.base)):  # added
                    self.count[i] = self.count[i<<1] + self.count[(i<<1)+1]

            def __apply(self, x, val):
                self.tree[x] = self.update_fn(self.tree[x], val*self.count[x])  # modified
                if x < self.base:
                    self.lazy[x] = self.update_fn(self.lazy[x], val)

            def __push(self, x):
                for h in reversed(xrange(1, x.bit_length())):
                    y = x>>h
                    if self.lazy[y] is not None:
                        self.__apply(y<<1, self.lazy[y])
                        self.__apply((y<<1)+1, self.lazy[y])
                        self.lazy[y] = None

            def update(self, L, R, h):  # Time: O(logN), Space: O(N)
                def pull(x):
                    while x > 1:
                        x >>= 1
                        self.tree[x] = self.query_fn(self.tree[x<<1], self.tree[(x<<1)+1])
                        if self.lazy[x] is not None:
                            self.tree[x] = self.update_fn(self.tree[x], self.lazy[x]*self.count[x])  # modified

                L += self.base
                R += self.base
                # self.__push(L)  # enable if range assignment
                # self.__push(R)  # enable if range assignment
                L0, R0 = L, R
                while L <= R:
                    if L & 1:  # is right child
                        self.__apply(L, h)
                        L += 1
                    if R & 1 == 0:  # is left child
                        self.__apply(R, h)
                        R -= 1
                    L >>= 1
                    R >>= 1
                pull(L0)
                pull(R0)

            def query(self, L, R):
                if L > R:
                    return None
                L += self.base
                R += self.base
                self.__push(L)
                self.__push(R)
                left = right = None
                while L <= R:
                    if L & 1:
                        left = self.query_fn(left, self.tree[L])
                        L += 1
                    if R & 1 == 0:
                        right = self.query_fn(self.tree[R], right)
                        R -= 1
                    L >>= 1
                    R >>= 1
                return self.query_fn(left, right)

        result = accu = 0
        sl = {}
        st = SegmentTree(len(nums))
        for i in xrange(len(nums)):
            j = sl[nums[i]] if nums[i] in sl else -1
            # sum(count(k, i)^2 for k in range(i+1)) - sum(count(k, i-1)^2 for k in range(i))
            # = sum(2*count(k, i-1)+1 for k in range(j+1, i+1))
            # = (i-j) + sum(2*count(k, i-1) for k in range(j+1, i+1))
            accu = (accu+((i-j)+2*max(st.query(j+1, i), 0)))%MOD
            result = (result+accu)%MOD
            st.update(j+1, i, 1)  # count(k, i) = count(k, i-1)+(1 if k >= j+1 else 0) for k in range(i+1)
            sl[nums[i]] = i
        return result


# Time:  O(n^2)
# Space: O(n)
# hash table
class Solution3(object):
    def sumCounts(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        MOD = 10**9+7
        result = 0
        for i in xrange(len(nums)):
            lookup = set()
            for j in reversed(xrange(i+1)):
                lookup.add(nums[j])
                result = (result+len(lookup)**2) % MOD
        return result

# 25 April 2024

# Time:  O(n)
# Space: O(1)

def read4(buf):
    global file_content
    i = 0
    while i < len(file_content) and i < 4:
        buf[i] = file_content[i]
        i += 1

    if len(file_content) > 4:
        file_content = file_content[4:]
    else:
        file_content = ""
    return i

# The read4 API is already defined for you.
# @param buf, a list of characters
# @return an integer
# def read4(buf):

class Solution(object):
    def __init__(self):
        self.__buf4 = [''] * 4
        self.__i4 = 0
        self.__n4 = 0

    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Maximum number of characters to read (int)
        :rtype: The number of characters read (int)
        """
        i = 0
        while i < n:
            if self.__i4 < self.__n4:  # Any characters in buf4.
                buf[i] = self.__buf4[self.__i4]
                i += 1
                self.__i4 += 1
            else:
                self.__n4 = read4(self.__buf4)  # Read more characters.
                if self.__n4:
                    self.__i4 = 0
                else:  # Buffer has been empty.
                    break

        return i

# 26 April 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def findPermutation(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        result = []
        for i in xrange(len(s)+1):
            if i == len(s) or s[i] == 'I':
                result += range(i+1, len(result), -1)
        return result

# 27 April 2024

# Time:  O(k^2 + r + c), r = len(rowConditions), c = len(colConditions)
# Space: O(k + r + c)

# topological sort
class Solution(object):
    def buildMatrix(self, k, rowConditions, colConditions):
        """
        :type k: int
        :type rowConditions: List[List[int]]
        :type colConditions: List[List[int]]
        :rtype: List[List[int]]
        """
        def topological_sort(conditions):
            adj = [[] for _ in xrange(k)]
            in_degree = [0]*k
            for u, v in conditions:
                u -= 1
                v -= 1
                adj[u].append(v)
                in_degree[v] += 1
            result = []
            q = [u for u in xrange(k) if not in_degree[u]]
            while q:
                new_q = []
                for u in q:
                    result.append(u)
                    for v in adj[u]:
                        in_degree[v] -= 1
                        if in_degree[v]:
                            continue
                        new_q.append(v)
                q = new_q
            return result

        row_order = topological_sort(rowConditions)
        if len(row_order) != k:
            return []
        col_order = topological_sort(colConditions)
        if len(col_order) != k:
            return []
        row_idx = {x:i for i, x in enumerate(row_order)}
        col_idx = {x:i for i, x in enumerate(col_order)}
        result = [[0]*k for _ in xrange(k)]
        for i in xrange(k):
            result[row_idx[i]][col_idx[i]] = i+1
        return result

# 28 April 2024

# Time:  O(l * (w + n)), l is the length of a word, w is the number of words, n is the length of target
# Space: O(n)

import collections


# optimized from Solution2
class Solution(object):
    def numWays(self, words, target):
        """
        :type words: List[str]
        :type target: str
        :rtype: int
        """
        MOD = 10**9+7
        dp = [0]*(len(target)+1)
        dp[0] = 1
        for i in xrange(len(words[0])):
            count = collections.Counter(w[i] for w in words)
            for j in reversed(xrange(len(target))):
                dp[j+1] += dp[j]*count[target[j]] % MOD
        return dp[-1] % MOD


# Time:  O(l * (w + n)), l is the length of a word, w is the number of words, n is the length of target
# Space: O(n)
import collections


class Solution2(object):
    def numWays(self, words, target):
        """
        :type words: List[str]
        :type target: str
        :rtype: int
        """
        MOD = 10**9+7
        # dp[i+1][j+1]: number of ways of target[0..j] using count[0..i].
        dp = [[0]*(len(target)+1) for _ in xrange(2)]
        for i in xrange(len(dp)):
            dp[i][0] = 1
        for i in xrange(len(words[0])):
            count = collections.Counter(w[i] for w in words)
            for j in reversed(xrange(len(target))):
                dp[(i+1)%2][j+1] = dp[i%2][j+1]+dp[i%2][j]*count[target[j]] % MOD
        return dp[(len(words[0]))%2][-1] % MOD

# 29 April 2024

# Time:  O(n)
# Space: O(1)

# dp
class Solution(object):
    def minIncrementOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        W = 3
        dp = [0]*W
        for i, x in enumerate(nums):
            dp[i%W] = min(dp[j%W] for j in xrange(i-W, i))+max(k-x, 0)
        return min(dp)

# 30 April 2024

# Time:  O(q * n)
# Space: O(1)

class Solution(object):
    def countPoints(self, points, queries):
        """
        :type points: List[List[int]]
        :type queries: List[List[int]]
        :rtype: List[int]
        """
        result = []
        for i, j, r in queries:
            result.append(0)
            for x, y in points:
                if (x-i)**2+(y-j)**2 <= r**2:
                    result[-1] += 1
        return result

# 01 May 2024

# Time:  O(n)
# Space: O(1)

# two pointers
class Solution(object):
    def longestSemiRepetitiveSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = left = prev = 0
        for right in xrange(len(s)):
            if right-1 >= 0 and s[right-1] == s[right]:
                left, prev = prev, right
            result = max(result, right-left+1)
        return result

# 02 May 2024

# Time:  O(n)
# Space: O(n)

# stack
class Solution(object):
    def removeStars(self, s):
        """
        :type s: str
        :rtype: str
        """
        result = []
        for c in s:
            if c == '*':
                result.pop()
            else:
                result.append(c)
        return "".join(result)

# 03 May 2024

# Time:  O(n)
# Space: O(1)

import operator


# bit manipulation
class Solution(object):
    def xorAllNums(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        return (reduce(operator.xor, nums1) if len(nums2)%2 else 0) ^ \
               (reduce(operator.xor, nums2) if len(nums1)%2 else 0)

# 05 May 2024

# Time:  O(nlogn)
# Space: O(n)

class Solution(object):
    def carPooling(self, trips, capacity):
        """
        :type trips: List[List[int]]
        :type capacity: int
        :rtype: bool
        """
        line = [x for num, start, end in trips for x in [[start, num], [end, -num]]]
        line.sort()
        for _, num in line:
            capacity -= num
            if capacity < 0:
                return False
        return True

# 06 May 2024

# Time:  O(1)
# Space: O(1)

# math
class Solution(object):
    def convertTemperature(self, celsius):
        """
        :type celsius: float
        :rtype: List[float]
        """
        return [celsius+273.15, celsius*1.80+32.00]

# 07 May 2024

# Time:  O(n) on average
# Space: O(1)

from random import randint

# Quick select solution.
class Solution(object):
    def minMoves2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def kthElement(nums, k):
            def PartitionAroundPivot(left, right, pivot_idx, nums):
                pivot_value = nums[pivot_idx]
                new_pivot_idx = left
                nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
                for i in xrange(left, right):
                    if nums[i] > pivot_value:
                        nums[i], nums[new_pivot_idx] = nums[new_pivot_idx], nums[i]
                        new_pivot_idx += 1

                nums[right], nums[new_pivot_idx] = nums[new_pivot_idx], nums[right]
                return new_pivot_idx

            left, right = 0, len(nums) - 1
            while left <= right:
                pivot_idx = randint(left, right)
                new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums)
                if new_pivot_idx == k:
                    return nums[new_pivot_idx]
                elif new_pivot_idx > k:
                    right = new_pivot_idx - 1
                else:  # new_pivot_idx < k.
                    left = new_pivot_idx + 1

        median = kthElement(nums, len(nums)//2)
        return sum(abs(num - median) for num in nums)

    def minMoves22(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        median = sorted(nums)[len(nums) / 2]
        return sum(abs(num - median) for num in nums)

# 08 May 2024

# Time:  O(m * n)
# Space: O(1)

class Solution(object):
    def placeWordInCrossword(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        def get_val(mat, i, j, transposed):
            return mat[i][j] if not transposed else mat[j][i]

        def get_vecs(mat, transposed):
            for i in xrange(len(mat) if not transposed else len(mat[0])):
                yield (get_val(mat, i, j, transposed) for j in xrange(len(mat[0]) if not transposed else len(mat)))

        for direction in (lambda x: iter(x), reversed):
            for transposed in xrange(2):
                for row in get_vecs(board, transposed):
                    it, matched = direction(word), True
                    for c in row:
                        if c == '#':
                            if next(it, None) is None and matched:
                                return True
                            it, matched = direction(word), True
                            continue
                        if not matched:
                            continue
                        nc = next(it, None)
                        matched = (nc is not None) and c in (nc, ' ')
                    if (next(it, None) is None) and matched:
                        return True
        return False


# Time:  O(m * n)
# Space: O(m * n)
class Solution2(object):
    def placeWordInCrossword(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        words = [word, word[::-1]]
        for mat in (board, zip(*board)):
            for row in mat:
                blocks = ''.join(row).split('#')
                for s in blocks:
                    if len(s) != len(word):
                        continue
                    for w in words:
                        if all(s[i] in (w[i], ' ') for i in xrange(len(s))):
                            return True
        return False

# 09 May 2024

# Time:  O(logn) = O(1)
# Space: O(1)

class Solution(object):
    # @return an integer
    def trailingZeroes(self, n):
        result = 0
        while n > 0:
            result += n / 5
            n /= 5
        return result

# 10 May 2024

# Time:  O(n^2) ~ O(n^3)
# Space: O(n^2)

import collections
import itertools


class Solution(object):
    def minAreaFreeRect(self, points):
        """
        :type points: List[List[int]]
        :rtype: float
        """
        points.sort()
        points = [complex(*z) for z in points]
        lookup = collections.defaultdict(list)
        for P, Q in itertools.combinations(points, 2):
            lookup[P-Q].append((P+Q) / 2)

        result = float("inf")
        for A, candidates in lookup.iteritems():
            for P, Q in itertools.combinations(candidates, 2):
                if A.real * (P-Q).real + A.imag * (P-Q).imag == 0.0:
                    result = min(result, abs(A) * abs(P-Q))
        return result if result < float("inf") else 0.0

# 11 May 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def maxAscendingSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = curr = 0
        for i in xrange(len(nums)): 
            if not (i and nums[i-1] < nums[i]):
                curr = 0
            curr += nums[i]
            result = max(result, curr)
        return result

# 12 May 2024

# Time:  O(m * n)
# Space: O(m * n)

# dp
class Solution(object):
    def minimumWhiteTiles(self, floor, numCarpets, carpetLen):
        """
        :type floor: str
        :type numCarpets: int
        :type carpetLen: int
        :rtype: int
        """
        dp = [[0]*(numCarpets+1) for _ in xrange(len(floor)+1)]  # dp[i][j] : min number of white tiles in the first i floors with j carpets
        for i in xrange(1, len(dp)):
            dp[i][0] = dp[i-1][0] + int(floor[i-1])
            for j in xrange(1, numCarpets+1):
                dp[i][j] = min(dp[i-1][j] + int(floor[i-1]), dp[max(i-carpetLen, 0)][j-1])
        return dp[-1][-1]

# 13 May 2024

# Time:  O(n^2)
# Space: O(n)

class Solution(object):
    def splitArray(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) < 7:
            return False

        accumulated_sum = [0] * len(nums)
        accumulated_sum[0] = nums[0]
        for i in xrange(1, len(nums)):
            accumulated_sum[i] = accumulated_sum[i-1] + nums[i]
        for j in xrange(3, len(nums)-3):
            lookup = set()
            for i in xrange(1, j-1):
                if accumulated_sum[i-1] == accumulated_sum[j-1] - accumulated_sum[i]:
                    lookup.add(accumulated_sum[i-1])
            for k in xrange(j+2, len(nums)-1):
                if accumulated_sum[-1] - accumulated_sum[k] == accumulated_sum[k-1] - accumulated_sum[j] and \
                   accumulated_sum[k - 1] - accumulated_sum[j] in lookup:
                    return True
        return False

# 14 May 2024

# Time:  O(n)
# Space: O(1)

# sliding window, two pointers
class Solution(object):
    def longestNiceSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = left = curr = 0
        for right in xrange(len(nums)):
            while curr&nums[right]:
                curr ^= nums[left]
                left += 1
            curr |= nums[right]
            result = max(result, right-left+1)
        return result

# 15 May 2024

# Time:  O(n + klogn)
# Space: O(1)

import heapq


class Solution(object):
    def minStoneSum(self, piles, k):
        """
        :type piles: List[int]
        :type k: int
        :rtype: int
        """
        for i, x in enumerate(piles):
            piles[i] = -x
        heapq.heapify(piles)
        for i in xrange(k):
            heapq.heappush(piles, heapq.heappop(piles)//2)
        return -sum(piles)

# 16 May 2024

# Time:  O(n)
# Space: O(n)

from collections import defaultdict

class Solution(object):
    def isRectangleCover(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: bool
        """
        left = min(rec[0] for rec in rectangles)
        bottom = min(rec[1] for rec in rectangles)
        right = max(rec[2] for rec in rectangles)
        top = max(rec[3] for rec in rectangles)

        points = defaultdict(int)
        for l, b, r, t in rectangles:
            for p, q in zip(((l, b), (r, b), (l, t), (r, t)), (1, 2, 4, 8)):
                if points[p] & q:
                    return False
                points[p] |= q

        for px, py in points:
            if left < px < right or bottom < py < top:
                if points[(px, py)] not in (3, 5, 10, 12, 15):
                    return False

        return True

# 17 May 2024

# Time:  O(m + n)
# Space: O(1)

class Solution(object):
    def countNegatives(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        result, c = 0, len(grid[0])-1
        for row in grid:
            while c >= 0 and row[c] < 0:
                c -= 1
            result += len(grid[0])-1-c
        return result

# 18 May 2024

# Time:  O(nlogn)
# Space: O(n)

class Solution(object):
    def arrayRankTransform(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        return map({x: i+1 for i, x in enumerate(sorted(set(arr)))}.get, arr)

# 29 May 2024

# Time:  precompute: O(max_n)
#        runtime:    O(s + logn)
# Space: O(max_n)

# combinatorics
FACT, INV, INV_FACT = [[1]*2 for _ in xrange(3)]
class Solution(object):
    def numberOfSequence(self, n, sick):
        """
        :type n: int
        :type sick: List[int]
        :rtype: int
        """
        MOD = 10**9+7
        def nCr(n, k):
            while len(INV) <= n:  # lazy initialization
                FACT.append(FACT[-1]*len(INV) % MOD)
                INV.append(INV[MOD%len(INV)]*(MOD-MOD//len(INV)) % MOD)  # https://cp-algorithms.com/algebra/module-INVerse.html
                INV_FACT.append(INV_FACT[-1]*INV[-1] % MOD)
            return (FACT[n]*INV_FACT[n-k] % MOD) * INV_FACT[k] % MOD
        
        result = 1
        total = cnt = 0
        for i in xrange(len(sick)+1):
            l = (sick[i] if i < len(sick) else n)-(sick[i-1] if i-1 >= 0 else -1)-1
            if i not in (0, len(sick)):
                cnt += max(l-1, 0)
            total += l
            result = (result*nCr(total, l))%MOD
        result = (result*pow(2, cnt, MOD))%MOD
        return result

# 30 May 2024

# Time:  O(n)
# Space: O(h)

import collections


"""
# Employee info
class Employee(object):
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates
"""
class Solution(object):
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        if employees[id-1] is None:
            return 0
        result = employees[id-1].importance
        for id in employees[id-1].subordinates:
            result += self.getImportance(employees, id)
        return result


# Time:  O(n)
# Space: O(w), w is the max number of nodes in the levels of the tree
class Solution2(object):
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        result, q = 0, collections.deque([id])
        while q:
            curr = q.popleft()
            employee = employees[curr-1]
            result += employee.importance
            for id in employee.subordinates:
                q.append(id)
        return result

# 31 May 2024

# Time:  O(n)
# Space: O(n)

import collections


class Solution(object):
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        def countSubtreeSumHelper(root, counts):
            if not root:
                return 0
            total = root.val + \
                    countSubtreeSumHelper(root.left, counts) + \
                    countSubtreeSumHelper(root.right, counts)
            counts[total] += 1
            return total

        counts = collections.defaultdict(int)
        countSubtreeSumHelper(root, counts)
        max_count = max(counts.values()) if counts else 0
        return [total for total, count in counts.iteritems() if count == max_count]

# 01 June 2024

# Time:  O(nlogm), m is the max of inventory, n is the size of inventory
# Space: O(1)

class Solution(object):
    def maxProfit(self, inventory, orders):
        """
        :type inventory: List[int]
        :type orders: int
        :rtype: int
        """
        MOD = 10**9+7
        def check(inventory, orders, x):
            return count(inventory, x) > orders
        
        def count(inventory, x):
            return sum(count-x+1 for count in inventory if count >= x)

        left, right = 1, max(inventory)
        while left <= right:
            mid = left + (right-left)//2
            if not check(inventory, orders, mid):
                right = mid-1
            else:
                left = mid+1
        # assert(orders-count(inventory, left) >= 0)
        return (sum((left+cnt)*(cnt-left+1)//2 for cnt in inventory if cnt >= left) +
                (left-1)*(orders-count(inventory, left)))% MOD

# 02 June 2024

# Time:  O(n)
# Space: O(1)

class Solution(object):
    def countSpecialSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        MOD = 10**9+7
        dp = [0]*3
        for x in nums:
            dp[x] = ((dp[x]+dp[x])%MOD+(dp[x-1] if x-1 >= 0 else 1))%MOD
        return dp[-1]

# 03 June 2024

# Time:  O(n^2)
# Space: O(n^2)

class Solution(object):
    def stoneGameV(self, stoneValue):
        """
        :type stoneValue: List[int]
        :rtype: int
        """
        n = len(stoneValue)
        prefix = [0]
        for v in stoneValue:
            prefix.append(prefix[-1] + v)

        mid = range(n)

        dp = [[0]*n for _ in xrange(n)]
        for i in xrange(n):
            dp[i][i] = stoneValue[i]

        max_score = 0
        for l in xrange(2, n+1):
            for i in xrange(n-l+1):
                j = i+l-1
                while prefix[mid[i]]-prefix[i] < prefix[j+1]-prefix[mid[i]]:
                    mid[i] += 1  # Time: O(n^2) in total
                p = mid[i]
                max_score = 0
                if prefix[p]-prefix[i] == prefix[j+1]-prefix[p]:
                    max_score = max(dp[i][p-1], dp[j][p])
                else:
                    if i <= p-2:
                        max_score = max(max_score, dp[i][p-2])
                    if p <= j:
                        max_score = max(max_score, dp[j][p])
                dp[i][j] = max(dp[i][j-1], (prefix[j+1]-prefix[i]) + max_score)
                dp[j][i] = max(dp[j][i+1], (prefix[j+1]-prefix[i]) + max_score)
        return max_score


# Time:  O(n^2)
# Space: O(n^2)
class Solution2(object):
    def stoneGameV(self, stoneValue):
        """
        :type stoneValue: List[int]
        :rtype: int
        """
        n = len(stoneValue)
        prefix = [0]
        for v in stoneValue:
            prefix.append(prefix[-1] + v)

        mid = [[0]*n for _ in xrange(n)]
        for l in xrange(1, n+1):
            for i in xrange(n-l+1):
                j = i+l-1
                p = i if l == 1 else mid[i][j-1]
                while prefix[p]-prefix[i] < prefix[j+1]-prefix[p]:
                    p += 1  # Time: O(n^2) in total
                mid[i][j] = p
        
        rmq = [[0]*n for _ in xrange(n)]
        for i in xrange(n):
            rmq[i][i] = stoneValue[i]

        dp = [[0]*n for _ in xrange(n)]
        for l in xrange(2, n+1):
            for i in xrange(n-l+1):
                j = i+l-1
                p = mid[i][j]
                max_score = 0
                if prefix[p]-prefix[i] == prefix[j+1]-prefix[p]:
                    max_score = max(rmq[i][p-1], rmq[j][p])
                else:
                    if i <= p-2:
                        max_score = max(max_score, rmq[i][p-2])
                    if p <= j:
                        max_score = max(max_score, rmq[j][p])
                dp[i][j] = max_score
                rmq[i][j] = max(rmq[i][j-1], (prefix[j+1]-prefix[i]) + max_score)
                rmq[j][i] = max(rmq[j][i+1], (prefix[j+1]-prefix[i]) + max_score)
        return dp[0][n-1]

# 04 June 2024

# Time:  O(n)
# Space: O(n)

import collections


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def removeZeroSumSublists(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        curr = dummy = ListNode(0)
        dummy.next = head
        prefix = 0
        lookup = collections.OrderedDict()
        while curr:
            prefix += curr.val
            node = lookup.get(prefix, curr)
            while prefix in lookup:
                lookup.popitem()
            lookup[prefix] = node
            node.next = curr.next
            curr = curr.next
        return dummy.next

# 05 June 2024

# Time:  O(n * 2^n), n is the size of the debt.
# Space: O(2^n)

import collections


class Solution(object):
    def minTransfers(self, transactions):
        """
        :type transactions: List[List[int]]
        :rtype: int
        """
        accounts = collections.defaultdict(int)
        for src, dst, amount in transactions:
            accounts[src] += amount
            accounts[dst] -= amount

        debts = [account for account in accounts.itervalues() if account]

        dp = [0]*(2**len(debts))
        sums = [0]*(2**len(debts))
        for i in xrange(len(dp)):
            bit = 1
            for j in xrange(len(debts)):
                if (i & bit) == 0:
                    nxt = i | bit
                    sums[nxt] = sums[i]+debts[j]
                    if sums[nxt] == 0:
                        dp[nxt] = max(dp[nxt], dp[i]+1)
                    else:
                        dp[nxt] = max(dp[nxt], dp[i])
                bit <<= 1
        return len(debts)-dp[-1]

# 06 June 2024

# Time:  O(n)
# Space: O(n)

class Solution(object):
    def findMaximums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def find_bound(nums, direction, init):
            result = [0]*len(nums)
            stk = [init]
            for i in direction(xrange(len(nums))):
                while stk[-1] != init and nums[stk[-1]] >= nums[i]:
                    stk.pop()
                result[i] = stk[-1]
                stk.append(i)
            return result

        left = find_bound(nums, lambda x: x, -1)
        right = find_bound(nums, reversed, len(nums))
        result = [-1]*len(nums)
        for i, v in enumerate(nums):
            result[((right[i]-1)-left[i])-1] = max(result[((right[i]-1)-left[i])-1], v)
        for i in reversed(xrange(len(nums)-1)):
            result[i] = max(result[i], result[i+1])
        return result

# 07 June 2024

# Time:  O(nlogn)
# Space: O(1)

class Solution(object):
    def minOperations(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def popcount(n):
            result = 0
            while n:
                n &= n-1
                result += 1
            return result

        result, max_len = 0, 1
        for num in nums:
            result += popcount(num)
            max_len = max(max_len, num.bit_length())
        return result + (max_len-1)

# 08 June 2024

# Time:  O(n^p) = O(1), n is the max number of possible moves for each piece, and n is at most 29
#                     , p is the number of pieces, and p is at most 4
# Space: O(1)

class Solution(object):
    def countCombinations(self, pieces, positions):
        """
        :type pieces: List[str]
        :type positions: List[List[int]]
        :rtype: int
        """
        directions = {"rook": [(0, 1), (1, 0), (0, -1), (-1, 0)],
                      "bishop": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
                      "queen" : [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]}
        all_mask = 2**7-1  # at most 7 seconds in 8x8 board
        def backtracking(pieces, positions, i, lookup):
            if i == len(pieces):
                return 1
            result = 0
            r, c = positions[i]
            r, c = r-1, c-1
            mask = all_mask
            if not (lookup[r][c]&mask):
                lookup[r][c] += mask  # stopped at (r, c)
                result += backtracking(pieces, positions, i+1, lookup)
                lookup[r][c] -= mask          
            for dr, dc in directions[pieces[i]]:
                bit, nr, nc = 1, r+dr, c+dc
                mask = all_mask  # (mask&bit == 1): (log2(bit)+1)th second is occupied
                while 0 <= nr < 8 and 0 <= nc < 8 and not (lookup[nr][nc]&bit):
                    lookup[nr][nc] += bit
                    mask -= bit
                    if not (lookup[nr][nc]&mask):  # stopped at (nr, nc)
                        lookup[nr][nc] += mask
                        result += backtracking(pieces, positions, i+1, lookup)
                        lookup[nr][nc] -= mask
                    bit, nr, nc = bit<<1, nr+dr, nc+dc
                while bit>>1:
                    bit, nr, nc = bit>>1, nr-dr, nc-dc
                    lookup[nr][nc] -= bit
            return result

        return backtracking(pieces, positions, 0, [[0]*8 for _ in range(8)])

# 09 June 2024

# Time:  O(n)
# Space: O(n)

import collections


class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        result = 0
        accumulated_sum = 0
        lookup = collections.defaultdict(int)
        lookup[0] += 1
        for num in nums:
            accumulated_sum += num
            result += lookup[accumulated_sum - k]
            lookup[accumulated_sum] += 1
        return result

# 10 June 2024

# Time:  O(m * n)
# Space: O(m + n)

# dfs
class Solution(object):
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
        if board[click[0]][click[1]] == 'M':
            board[click[0]][click[1]] = 'X'
            return board
        stk = [click]
        while stk:
            r, c = stk.pop()
            cnt = 0
            adj = []
            for dr in xrange(-1, 2):
                for dc in xrange(-1, 2):
                    if dr == dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < len(board) and 0 <= nc < len(board[r])):
                        continue
                    if board[nr][nc] == 'M':
                        cnt += 1
                    elif board[nr][nc] == 'E':
                        adj.append((nr, nc))
            if cnt:
                board[r][c] = chr(cnt + ord('0'))
                continue
            board[r][c] = 'B'
            for nr, nc in adj:
                board[nr][nc] = ' '
                stk.append((nr, nc))
        return board


# Time:  O(m * n)
# Space: O(m + n)
# dfs
class Solution2(object):
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
        if board[click[0]][click[1]] == 'M':
            board[click[0]][click[1]] = 'X'
            return board
        q = [click]
        while q:
            new_q = []
            for r, c in q:
                cnt = 0
                adj = []
                for dr in xrange(-1, 2):
                    for dc in xrange(-1, 2):
                        if dr == dc == 0:
                            continue
                        nr, nc = r+dr, c+dc
                        if not (0 <= nr < len(board) and 0 <= nc < len(board[r])):
                            continue
                        if board[nr][nc] == 'M':
                            cnt += 1
                        elif board[nr][nc] == 'E':
                            adj.append((nr, nc))
                if cnt:
                    board[r][c] = chr(cnt + ord('0'))
                    continue
                board[r][c] = 'B'
                for nr, nc in adj:
                    board[nr][nc] = ' '
                    new_q.append((nr, nc))
            q = new_q
        return board

# 11 June 2024

# Time:  O(n + mlogm), m is the number of rides
# Space: O(n)

class Solution(object):
    def maxTaxiEarnings(self, n, rides):
        """
        :type n: int
        :type rides: List[List[int]]
        :rtype: int
        """
        rides.sort()
        dp = [0]*(n+1)
        j = 0
        for i in xrange(1, n+1):
            dp[i] = max(dp[i], dp[i-1])
            while j < len(rides) and rides[j][0] == i:
                dp[rides[j][1]] = max(dp[rides[j][1]], dp[i]+rides[j][1]-rides[j][0]+rides[j][2])
                j += 1
        return dp[-1]

