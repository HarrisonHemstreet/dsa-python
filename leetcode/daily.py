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

