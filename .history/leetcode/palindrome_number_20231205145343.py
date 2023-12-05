from typing import List
class PalindromeNumber:
    def run(self, num: int) -> bool:
        if num % 2 != 0:
            return False

        num_list: List[str] = list(f'{num}')
        rev_num: int = int("".join(num_list[::-1]))

        return True if num == rev_num else False