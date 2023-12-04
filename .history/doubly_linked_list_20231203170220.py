# contains a head, tail and length property.
# each node points to another node or null.
class Node:
    def __init__(self, val, next = None, prev = None):
        self.val: int = val
        self.next: Node | None = next
        self.prev: Node | None = prev

    def __str__(self):
        return (f"""
            Node(
                val: {self.val},
                prev: {self.prev},
                next: {self.next}
            )
        """)

class DoublyLinkedList:
    def __init__(self, head = None, tail = None, length = 0):
        self.head: Node | None = head
        self.tail: Node | None = tail
        self.length: int = length

    def __str__(self):
        return f"DoublyLinkedList: {self.head}"

    # O(1)
    def push(self, val: int):
        new_node = Node(val)

        if self.head == None:
            self.head = new_node
            self.tail = new_node
            self.head.next = self.tail
            self.tail.prev = self.head
        elif self.tail is not None:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1
        return self.head

    def get_len(self):
        return self.length

    # O(n)
    def find(self, val: int):
        curr_node = self.head
        while curr_node is not None:
            if curr_node.val == val:
                return curr_node
            else:
                curr_node = curr_node.next
        if curr_node == None:
            return f"Could not find {val}"

    # O(1) / O(n)
    def pop(self):
        curr_node = self.head
        while curr_node.next.next is not None:
            curr_node = curr_node.next
        curr_node.next = None
        self.tail = curr_node
        self.length -= 1
        return self.head

    # O(1)
    def shift(self, val: int):
        new_node = Node(val)
        new_node.next = self.head
        self.head = new_node
        self.length += 1
        return self.head

    # O(1)
    def unshift(self):
        if self.head == None:
            return "LIST IS EMPTY!"
        self.head = self.head.next
        self.length -= 1
        return self.head

    # take in an "index" and return the node located there
    def get(self, index):
        curr_node = self.head
        i = index
        while i > 1:
            i -= 1
            curr_node = curr_node.next
        return curr_node

    # take in value and an index of the node we are changing the value to
    def set(self, val, index):
        curr_node = self.head
        i = index
        while i > 1:
            i -= 1
            curr_node = curr_node.next
        curr_node.val = val
        return curr_node
    
    # like set, but inserts a new node instead of changing the value
    def insert(self, val, index):
        curr_node = self.head
        i = index
        while i > 2:
            i -= 1
            curr_node = curr_node.next
        node_after = curr_node.next
        curr_node.next = Node(val)
        curr_node.next.next = node_after
        self.length += 1
        return curr_node.next

    # like insert but the opposite
    def remove(self, index):
        curr_node = self.head
        i = index
        while i > 2:
            i -= 1
            curr_node = curr_node.next
        to_remove = curr_node.next
        curr_node.next = curr_node.next.next
        return to_remove

    def reverse(self):
        if self.head is None or self.tail is None:
            return self
        
        prev = None
        curr = self.head

        while curr is not None:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next

        accu = prev
        self.head = accu
        return prev

s_list = DoublyLinkedList()
s_list.push(1)
s_list.push(2)
s_list.push(3)
s_list.push(4)
s_list.push(5)
s_list.push(6)
s_list.push(7)
s_list.push(8)
# print("find:", s_list.find(8))
# print("poped:", s_list.pop())
# print("find:", s_list.find(8))
# print("shift:", s_list.shift(0))
# print("unshift:", s_list.unshift())
# print(s_list.get_len())
# print("get:", s_list.get(4))
# print("set:", s_list.set(100, 4))
# print("insert:", s_list.insert(200, 5))
# print("remove:", s_list.remove(5))
print("s_list:", s_list)
# print("s_list reverse:", s_list.reverse())
