class ListNode:
    def __init__(self, val = 0, next = None):
        self.val: int | None = val
        self.next: self = next

    def __repr__(self):
        return "ListNode(val=" + str(self.val) + ", next={" + str(self.next) + "})"
    
    def __eq__(self, other):
        if isinstance(other, ListNode):
            current_self = self
            current_other = other
            while current_self is not None and current_other is not None:
                if current_self.val != current_other.val:
                    return False
                current_self = current_self.next
                current_other = current_other.next
            return current_self is None and current_other is None
        return False

print(ListNode(1, ListNode(4, ListNode(5, None))))