from typing import List

class LongestCommonPrefix:
    def run(self, strs: List[str]) -> str:
        l_strs = len(strs)

        if l_strs == 1:
            return strs[0]
        elif l_strs == 0:
            return ""

        first: str = strs[0]
        l_first: int = len(first)
        longest: int = 0
        l_longest: List[int] = []

        for w in strs[1:]:
            for k, char in enumerate(w):
                if k > l_first - 1:
                    break
                
                if char == first[k]:
                    longest += 1
                else:
                    break
            l_longest.append(longest)
            longest = 0

        return first[0:min(l_longest)]

"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

 

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
 

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lowercase English letters.
"""