#1. algorithm 틀 짜기
class Node:
    def __init__(self, data):
        if not data:
            raise ValueError("nothing is imported")
        self.data = data
        self.next = None
        self.rear = None
        
class MakeNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("nothing is imported")
        self.lists = {}
        
        for i, value in enumerate(args, start=1):
            current = Node(value)
            self.lists[i] = current

        print(current)

MakeNode(3,4,5,6,7)

     

        
