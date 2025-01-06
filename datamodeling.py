# 1. Array
class ArrayModel:
    def __init__(self, element, list_name):
        self.element = element
        self.list_name = list_name
        self.lists = {}

    def add_to_list(self):
        if self.list_name not in self.lists:
            self.lists[self.list_name] = []
        self.lists[self.list_name].append(self.element)

    def get_list(self):
        return self.lists.get(self.list_name, [])


model1 = ArrayModel(5, "my_list")
model1.add_to_list()
print(model1.get_list())


# 2. Linked List_1
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        # node -> (data + pointer)


class LinkedList:
    def __init__(
        self, *args
    ):  # args == "가변인자"(여기서 args == "list"형식으로 보면은 되겠군)
        if not args:
            raise ValueError("least one element is required")

        self.head = Node(args[0])
        current = self.head  # current = self.head(현재 node == "head")

        for data in args[1:]:
            current.next = Node(data)
            current = current.next

        self.tail = current  # current == pointer.
        """마지막 node를 list의 끝으로 나타내는 tail로 설정."""

    def display_list(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("none")


# 2-1 Linked List_2 (원형 연결 list)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList1:
    def __init__(self, *args):
        if not args:
            raise ValueError("least one element is required")

        self.head = Node(args[0])
        current = self.head

        for data in args[1:]:
            current.next = Node(data)
            current = current.next

            self.tail = current
            self.tail.next = self.head

    def display_list(self):
        current = self.head
        while True:
            print(current.data, end=" -> ")
            current = current.next
            if current == self.head:
                break
        print("Back to Head ->", current.data)


# 2-2 Linked List_3(이중 연결 list)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.before = None


class LinkedList2:
    def __init__(self, *args):
        if not args:
            raise ValueError("least one element is required")

        self.head = Node(args[0])  # head 설정 완료.
        current = self.head

        for data in args[1:]:
            new_node = Node(data)
            current.next = new_node  # 현재 node next를 새 노드로 연결.
            new_node.before = current  # 새 노드의 이전을 현재 node로 연결.
            current = new_node  # 그 후 current를 새 노드로 갱신.

        self.tail = current

    def display_list_forward(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next

        print("None")

    def display_list_backward(self):
        current = self.tail
        while current:
            print(current.data, end=" -> ")
            current = current.before
        print("None")


# 3. Stack
class Stacker:

    def __init__(self, name, *args):
        self.name = name
        self.lists = {}

        if not args:
            raise ValueError("nothing imported")

        self.lists[self.name] = list(args)

    def pushing(self, value):
        self.lists[self.name].append(value)

    def pop(self):
        if self.lists[self.name]:
            return self.lists[self.name].pop()
        else:
            raise IndexError("empty stack")

    def peek_top(self):
        if self.lists[self.name]:
            return self.lists[self.name]
        else:
            raise IndexError("empty stack")


# 4. Queue
class Queue:

    def __init__(self, name, args):
        self.name = name
        self.lists = {}

        if not args:
            raise ValueError("nothing imported")

        self.lists[self.name] = list(args)

    def enqueque(self, value):
        self.lists[self.name].append(value)

    def Dequeue(self):
        if self.lists[self.name]:
            return self.lists[self.name].pop(0)
        else:
            raise IndexError("empty")

    def show(self):
        if self.lists[self.name]:
            return self.lists[self.name]
        else:
            raise IndexError("empty queue")


# 4-1. Linear Queue
class Queue1:

    def __init__(self, name, args):
        self.name = name
        n = len(args)
        mylist = list(args) + [None] * 3

        self.lists = {}

        if not args:
            raise ValueError("nothing imported")

        self.lists[self.name] = mylist

    def make_que(self, value):
        self.lists[self.name].append(value)

    def dequeue(self):
        if self.lists[self.name]:
            rvalue = self.lists[self.name].pop(0)
            self.lists[self.name].append(None)
            return rvalue
        else:
            raise IndexError("empty")

    def show(self):
        return self.lists[self.name]


# 4-2 circle queue
class Cqueue:

    def __init__(self, name, size, *args):
        self.name = name
        self.max_size = size + 3
        self.lists = [None] * self.max_size
        self.front = 0
        self.rear = -1  # -1로 지정하여 마지막 element가 존재하는 곳의 번호를 indexing.
        self.count = 0

        for arg in args:
            self.enqueue(arg)

    def enqueue(self, value):
        if self.count == self.max_size:
            self.lists[self.name].append(None)
        elif self.count >= self.max_size:
            raise OverflowError("queue is full")

        self.rear = (self.rear + 1) % self.max_size
        """etc:: when rear is four, it makes it go to zero"""
        self.lists[self.rear] = (
            value  # rear point 증가 후 실행이므로, 그 바로 다음자리에 원소를 input.
        )
        self.count += 1

    def dequeue(self):
        if self.count == 0:
            raise IndexError("empty")

        dvalue = self.lists[self.front]
        self.lists[self.front] = None
        self.front = (self.front + 1) % self.max_size
        self.count -= 1
        return dvalue

    def show(self):
        if self.count == 0:
            return []

        if self.rear >= self.front:
            return self.lists[self.front : self.rear + 1]
        else:
            return self.lists[self.front :] + self.lists[: self.rear + 1]
        # 이건 순환한 큐이므로 나누어져서 들어가 삽입되어 있다. 그러므로 앞의 것과 뒤의 것을 합쳐서 원소들을 추출해낸다.


# 5.Hash table


class Hashtable:

    def __init__(self, name, *args):
        self.name = name
        # n = range(len(args)) <enumerate> 함수의 존재로 n = 이런 식으로 할 필요가 없다.
        self.lists = {}

        for i, value in enumerate(args):
            hash_key = self.hashfunc(value, len(args))
            self.lists[hash_key] = value

    def hashfunc(self, value, size):
        return hash(value) % size

    def show_table(self):

        for key, value in self.lists.items():
            print(f"key: {key} -> value: {value}")


# 하지만 이 function은 hash함수에 크게 의존한다는 단점이 존재.

# 5. Hash table(if hash not)::


class MyHashTable:

    def __init__(self, name, *args):
        self.name = name
        self.lists = {}
        if not args:
            raise ValueError("nothing is inserted")

        for i, value in enumerate(args, start=1):
            self.lists[i] = value  # self.lists[i] 이렇게 순서대로 value값으로 저장.
            # dictionary == {key-value}값 형식으로 저장되어 있으므로 딕셔너리에서 data에 접근하기 위한 식별자 자체가 key

    def get(self, key):
        return self.lists.get(key, "key not found")

    # get 메서드 == 첫번쨰 인자가 찾을 키. 두번째 인자는 못찾을 경우 반환할 기본값.

    def show_table(self):
        for (
            key,
            value,
        ) in self.lists.items():  # 키와 값을 쌍으로 반환.(모든 항목에 대해 iterate)
            print(f"key: {key} -> value: {value}")


# 6. Graph

"""directed node"""


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None  # 이건 다음 노드로의 연결을 정의하고, 기본값을 none.


class MakeNode:
    def __init__(self, *args):
        self.lists = {}

        if not args:
            raise ValueError("nothing is imported")

        for i, value in enumerate(args, start=1):
            node = Node(value)
            self.lists[i] = node

    def connect(self):
        nodekey = list(self.lists.keys())
        # self.lists는 딕셔너리이므로 key값을 가지고 있고, 그 key갑들만 가져와서 list로 생성.

        for i in range(1, len(nodekey)):
            self.lists[i].next = self.lists[i + 1]
        # 반복문으로 node들을 전부 한방향으로 연결.


class Graph:
    def __init__(self, make_node):
        self.start_node = make_node.lists[1]

    def display(self):
        current = self.start_node
        # while 반복문을 use해서 next로 쭉 가서 마지막까지 간다.
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("nothing")  # 더이상 반복할 것이 없는 경우에서 이 list형식을 표시.


# 6-1 Graph -1


class Node1:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.before = None


class MakeNode2:
    def __init__(self, *args):
        self.lists = {}

        if not args:
            raise ValueError("nothing is implemented")

        for i, value in enumerate(args, start=1):
            node = Node1(value)
            self.lists[i] = node
        print("created nodes:", self.lists)

    def connect_nodes(self, from_idx, to_idx):
        """node index -> connect them"""
        if from_idx in self.lists and to_idx in self.lists:
            self.lists[from_idx].next = self.lists[to_idx]
            self.lists[to_idx].before = self.lists[from_idx]
            print("node1 {from_idx} -> node2 {to_idx}")
        else:
            raise IndexError("error occured")


# 7. Tree

import random


class Node:
    def __init__(self, data):
        if not data:
            raise ValueError("error")
        self.data = data
        self.neighbors = []


class MakeTreeNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("Error")
        self.lists = []
        self.root = None

        for value in args:
            tree_node = Node(value)
            self.lists.append(tree_node)
            # 이런 식으로 한 경우에 Tree로 value 인자들을 받아서 구현한다.

    def ChooseRoot(self):
        n = len(self.lists)
        if n == 0:
            raise ValueError("Error")

        root_index = random.randint(0, n - 1)
        self.root = self.lists[root_index]

        return self.root

    def CreateLayers(self, *args):
        if self.root is None:
            raise ValueError("Error: no root node")

        layers = [[self.root]]  # 이중 list 구조
        remain_nodes = [node for node in self.lists if node != self.root]
        # remain_nodes에는 self.root 제외되어 있음.

        if sum(args) != len(remain_nodes):
            raise ValueError("Error : sum is not properly defined")

        layers = []
        start_index = 0

        for count in args:  # count == 현재 층에 포함딜 node 개수
            if start_index < len(remain_nodes):
                end_index = start_index + count
                if end_index > len(remain_nodes):
                    end_index = len(remain_nodes)

                layers_nodes = remain_nodes[start_index:end_index]
                # start_index부터 end_index를 해서 이런 식으로 연결을 해준다.
                layers.append(layers_nodes)

                start_index = end_index

        return layers

    # hasattr == "변수(object)의 존재 여부를 확인하는 함수이다.(bool형태로 이루어진다.)"
    def ConnectLayers(self, layers):
        for i in range(len(layers) - 1):
            for parent in layers[i]:  # layers[i]의 각 노드가 부모 node로 취급된다.
                for child in layers[i + 1]:
                    parent.neighbors.append(child)


tree = MakeTreeNode(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # 10개의 노드 생성
tree.ChooseRoot()
layers = tree.CreateLayers(2, 3, 4)  # 층을 2-3-4로 생성
tree.ConnectLayers(layers)  # 층 연결

for i, layer in enumerate(layers):
    print(f"Layer {i + 1}:")
    for node in layer:
        neighbors = [neighbor.data for neighbor in node.neighbors]
        print(f"Node {node.data} -> Neighbors: {neighbors}")

# Wiki docs Tree(binary)


class Node:
    def __init__(self, data):
        self.data = data  # data 초기화.
        self.left = None
        self.right = None


# 여기서는 self를 리스트로 하지 않고 왼쪽 오른쪽을 정의하였다(자식 node(좌), 자식 node(우))


class BSTree:
    def __init__(self):
        self.root = None  # Tree의 root node 지정.

    """
    Tree class는 위로 갈수록, 즉 부모 node가 자식 Node보다 커야 하므로 이런 식으로 동작.
    """

    def __contains__(self, data):
        node = self.root  # 탐색을 시작할 node를 root node로 설정.
        while node:  # node 존재전까지 repeat 탐색.
            if node.data == data:
                return True
            elif node.data > data:
                node = node.left
            else:
                node = node.right
        return False

    # root node부터 탐색 start.
    """
    같으면 True, 없으면 false, 그리고 값이 작으면 left, 크면 right로 이동.
    """

    def inorder(self):
        def _inorder(node):
            if not node:
                return
            _inorder(node.left)
            res.append(node.data)
            _inorder(node.right)

        res = []  # res = 결과를 저장할 빈 list생성.
        _inorder(self.root)  # tree node부터 순회 start.
        return res

    def _insert(self, node, data):
        if node is None:  # node를 저장할 자리가 빈 것.
            return Node(data)
        if node.data > data:
            node.left = self._insert(node.left, data)
        else:
            node.right = self._insert(node.right, data)
        return node

    def insert(self, data):
        self.root = self._insert(self.root, data)
        # 위의 함수를 받아서 root부터 start하여 적절한 위치에 data insert하는 method.

    def get_leftmost(self, node):
        # get left에서 제일 작은 값.(즉, 끝까지 아래로 내려가서 가장 왼쪽에 있는 node return.)
        while node.left:
            node = node.left
        return node

    def _delete(self, node, data):
        if node is None:  # del 할 값이 tree에 존재 x.
            return
        if node.data > data:
            node.left = self._delete(node.left, data)
        elif node.data < data:
            node.right = self._delete(node.right, data)
        else:
            if node.left and node.right:
                leftmost = self.get_leftmost(node.right)
                node.data = leftmost.data
                node.right = self._delete(node.right, leftmost.data)
                return node
            if node.left:
                return node.left
            else:
                return node.right
        return node

    def delete(self, data):
        self.root = self._delete(self.root, data)
        # 위의 것과 비슷하게 가져와서 루트부터 start.


if __name__ == "__main__":
    bst = BSTree()
    for x in (8, 3, 10, 1, 6, 9, 12, 4, 5, 11):
        bst.insert(x)

# 8. Heap


class Node3:
    def __init__(self, data):
        if not data:
            raise ValueError("error occured")
        self.data = data
        self.border = []


class HeapNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("args are not imported")
        self.lists = []  # self.lists 라는 list를 정의
        self.root = None  # self.root 값을 만들고 none을 기본값으로 두기

        for value in args:
            heap_node = Node3(
                value
            )  # args의 value개수만큼 heap_node -> self.lists 안에 추가
            self.lists.append(heap_node)

    def MakeHeapNode(self):
        n = len(self.lists)
        if n == 0:
            raise ValueError("error occured")
        self.root = max(self.lists, key=lambda x: x.data)
        # lambda == 파이썬 익명 함수 define.(x == Node3)
        return self.root

    def CreateNode(self, *args):
        if self.root is None:
            raise ValueError("error occured")

        layers = [[self.root]]  # 이중 list를 생성해놓음.
        remain = [node for node in self.lists if node != self.root]

        if sum(args) != len(remain):
            raise ValueError("error")
        """lambda 함수는 값을 바로 반환하고 lambda(인자, 함수) == a로 할당하여 불러온다.
        (한줄로 간단하게 정의할 때 use.)"""
        layers = []
        start = 0

        for count in args:
            if start < len(remain):
                end_index = start + count
                if end_index > len(remain):
                    end_index = len(remain)

                layers_nodes = sorted(
                    remain[start:end_index], key=lambda x: x.data, reverse=True
                )

                layers.append(layers_nodes)
                start = end_index

        return layers  # 층 쌓기는 완료.

    def Connect(self, layers):
        for i in range(self, layers):
            for parent in layers[i]:
                for child in layers[i + 1]:
                    parent.neighbors.append(child)

    def SelfSort(self):
        queue = [self.root]
        sorted_values = []

        while queue:
            now = queue.pop(0)
            sorted_values.append(now.data)

            for neighbor in now.neighbors:
                queue.append(neighbor)

        return sorted_values

    def AddHeap(self, value):
        # new node 생성.
        new_node = Node3(value)
        self.lists.append(new_node)
        self.heapify_up(len(self.lists) - 1)

    def heapify_up(self, index):
        # root 노드와 indexing 비교
        if index == 0:
            return  # 재귀함수이므로 이렇게 작동이 가능하다.
        parent_index = (index - 1) // 2
        if self.lists[index].data > self.lists[parent_index].data:
            # parent index를 계산하여 indexing을 하여 data크기 비교.
            self.lists[index], self.lists[parent_index] = (
                self.lists[parent_index],
                self.lists[index],
            )
            # 크기가 오름차순으로 되어 있는 경우가 아니라면 교체 + 재귀적으로 끊임없이 repeat.
            self.heapify_up(parent_index)


"""
Sorted() == key(sorted) 함수에서 use되는 속성값.
lambda(X) == 익명함수 -> 즉, 일회성으로 넣고 함수를 저장하지 않음으로 불필요한 연산을 최소화시키는것이 목적.
"""

#######################################################################################

# Wikidocs data modeling


class Return:
    def findout(self, *args):
        if not args:
            raise ValueError("Error Occured")

        mylists = list(args)
        n = len(mylists)
        if n < 2:
            raise ValueError("only one is implied")

        mylists.sort(reverse=True)
        print(f"first{mylists[0]}, second{mylists[1]}")


class FindReverse:
    def findout(self, s):
        mylist = list(s)

        # data 정제
        s = "".join(e.lower() for e in s if e.isalnum())
        # 입력 문자열 s의 각 문자를 e로 가져오고, isalnum은 문자 혹은 숫자를 제외하고 전부 제거. + 소문자로 change.
        # .join == 필터링된 모든 문자열들이 전부 하나의 문자열로 결합.

        if mylist == mylist[::1]:
            return f"good"
        else:
            raise ValueError("error")


class Sort:
    def findandsort(self, args):
        if not args:
            raise ValueError("nothing imported")
        mylist = list(args)
        a = mylist.count(0)
        b = mylist.count(1)
        list2 = ([0] * a) + ([1] * b)
        return list2


# 이중 repeat으로 make.


def find_sub_Array(a: list[int], s: int) -> list[int]:
    # 이런 식으로 함수를 define (매개변수와 반환 의 type를 지정하는 것)
    # a, s) --  매개변수 . 반환값 -.> int형 의 리스트
    for i in range(len(a)):
        sub_total: int = 0  # 부분 배열의 합을 0으로 초기화.
        for j in range(i, len(a)):  # i부터 끊임없이 반복.
            sub_total += a[j]
            if sub_total == s:
                return [i + 1, j + 1]  # index return.
    return [-1]


# 이중 repeat 사용하지 않는 code 구현.


def find_sub_array_1(a: list[int], s: int) -> list[int]:

    left: int = 0
    sub_total: int = 0
    for right in range(len(a)):
        sub_total += a[right]
        # a[right] indexing을 use해서 끊임없이 반복.
        while left < right and sub_total > s:
            sub_total -= a[left]
            left += 1

        if sub_total == s:
            return [left + 1, right + 1]
    return [-1]


# AVL tree

"""
1. 노드의 height == 노드에서 가장 깊은 노드까지의 길이(하나의 직선)거쳐가는 경우 존재 x
2. node balance factor = right left 높이의 차이 
"""


class AVLnode(Node):
    def __init__(self, data):
        super().__init__(data)  # super() - 부모 class Node method 호출 use.
        self.height = 0  # node height 속성을 초기화(기본 높이를 0으로 define)


class AVLTree(BSTree):

    def get_height(self, node):
        if node is None:  # node == 높이를 계산하고자 하는 node.
            return -1  # -1은 node가 비어 있는 경우에 계산을 쉽게 구별할 수 있게 해줌.

        left_height = node.left.height if node.left else -1
        # node 왼쪽 자식이 존재하면, 그 자식의 높이를 가져옴. 그렇지 않으면 -1
        right_height = node.right.height if node.right else -1
        # node 오른쪽 자식도 같은 방식으로 높이를 계산함.
        return 1 + max(left_height, right_height)
        # 현재 노드 높이는 1 + (왼쪽, 오른쪽 자식의 높이 중 큰 값을 반환.)

    def get_balance(self, node):
        return self.get_height(node.left) - self.get_height(node.right)

    # 왼쪽 서브트리 높이에서 오른쪽 서브트리 높이를 빼서 균형인수를 계산.
    # get_balance 메서드를 use해서 0, -1, 1까지는 두고, -2이하 혹은 2이상값이 나오는 경우를 return
    # 하여 다시 회전연산이 필요하다는 것을 뜻함.

    def _insert(self, node, data):
        # node = 현재 탐색 중 node.(삽입 위치를 Find.) + data(tree) 삽입할 data 값.
        if node is None:
            return AVLnode(data)
        # node가 none인지 확인해야 하고, none 인 경우에 새로운 AVLnode 인스턴스 생성, 반환.
        if node.data > data:
            # 현재 node data가 삽입할 data보다 큰 경우에 왼쪽 서브트리에 삽입.(insert 메서드 호출)
            node.left = self._insert(node.left, data)
        else:
            node.right = self._insert(node.right, data)
            # 그렇지 않은 경우에 오른쪽 node_right 에 삽입한다.

        node.height = self.get_height(node)
        # node 높이 update
        node = self.balancing(node)
        # self.balancing을 호출하여 현재 node의 균형을 유지함.
        return node

    def _delete(self, node, data):
        if node is None:
            return
        if node.data > data:
            node.left = self._delete(node.left, data)

        elif node.data < data:

            node.right = self._delete(node.right, data)
        else:

            if node.left and node.right:
                leftmost = self.get_leftmost(node.right)
                node.data = leftmost.data
                node.right = self._delete(node.right, leftmost.data)
                return node
            if node.left:
                return node.left
            else:
                return node.right

        node.height = self.get_height(node)
        node = self.balancing(node)
        return node

    def rotate_right(self, node):
        child = node.left  # child를 node.left에 지정되어 있는 node에 선언.
        node.left, child.right = child.right, node
        ######### == 여기까지가 회전시키는 동작이라고 일컬을 수 있다.
        # node.left를 child.right로 설정하여 child의 오른쪽 자식이 node 좌 자식.
        node.height = self.get_height(node)
        child.height = self.get_height(child)
        return child

    def rotate_left(self, node):
        child = node.right  # 오른쪽 자식이 무거운 경우 왼쪽 회전.
        node.right, child.left = child.left, node
        node.height = self.get_height(node)
        child.height = self.get_height(child)
        return child

    def rotate_right_left(self, node):
        child = node.right  # 무거운 경우(오른쪽 자식보다 왼쪽이 무거운 경우)
        grandchild = child.left

        node.right, child.left = grandchild.left, grandchild.right
        grandchild.left, grandchild.right = node, child
        child.height = self.get_height(child)
        node.height = self.get_height(node)
        grandchild.height = self.get_height(grandchild)

        return grandchild

    def rotate_left_right(self, node):
        child = node.left
        # child -> 변수에 저장, 그리고 그 후에 그 child를 저장하여 나중에 root node로 변경.
        grandchild = child.right
        # grandchild 변수에 저장한 후 이걸 회전을 시키는 경우에 중심으로 use.
        child.right, node.left = grandchild.left, grandchild.right
        grandchild.left, grandchild.right = child, node
        # 이 위의 두줄의 코드로 자식관계를 재조정.
        # 파이썬에서 두가지 작업 동시 수행 튜플 할당 방식
        child.height = self.get_height(child)
        node.height = self.get_height(node)
        grandchild.height = self.get_height(grandchild)
        # 높이 계산하고 그 후에 위에 정의한 함수를 바탕으로 높이 재조정.

        return grandchild

    def balancing(self, node):
        balance_factor = self.get_balance(node)
        if balance_factor == 2:
            if self.get_balance(node.left) >= 0:
                node = self.rotate_right(node)
            else:
                node = self.rotate_left_right(node)

        elif balance_factor == -2:
            if self.get_balance(node.right) <= 0:
                node = self.rotate_left(node)
            else:
                node = self.rotate_right_left(node)
        return node


if __name__ == "__main__":  # 시작점을 정의.(define)

    def level_order(tree):
        q = [tree.root]
        # q = 리스트 생성, 리스트의 첫번째 요소로 tree의 root.node를 추가함.
        # node 저장하는 자료구조(queue) BFS(너비 우선탐색)으로 노드 탐색.
        res = []
        # 방문한 각 node의 data를 저장하는 데 use.
        while q:  # q에 노드가 남아 있는 동안 repeat.
            node = q.pop()  # 방문할 node 꺼내고 요소 반환.
            res.append((node.data, node.height, tree.get_balance(node)))
            # 튜플 형태로 res list에 추가함(균형 인수)
            if node.left:  # 존재하는 경우에 따라서 추가하여 다음순서에서 방문.
                q.append(node.left)
            if node.right:
                q.append(node.right)

        return res

    def get_data(msg):  # msg - user에게 보여줄 메시지 전달 문자열.
        print(msg, end=">>> ")  # end를 이런 식을 정의하여 입력 유도.
        data = input()  # 사용자 입력을 받아서 data 변수에 str 형태로 저장.
        if data.isdigit():
            return int(
                data
            )  # 숫자인지 확인하여 반환.(숫자가 아니면 false.-> none 반환.)
        return None

    tree = AVLTree()
    # AVLtree 클래스의 인스턴스 Tree 생성.
    while True:
        # 무한 루프 start.
        menu = """
실행할 명령어를 선택하세요.

[0] AVL 트리의 상태 출력 [(노드 값, 높이, 균형 지수), ...]
[1] 노드 추가
[2] 노드 삭제
[3] 노드 검색
[9] 끝내기
"""
        # 사용자와의 상호작용을 지속적으로 처리함.
        print(menu, end=" >>> ")
        # menu 문자열 출력.
        command = int(input())
        # command 변수에 저장하고, 문자열로 입력받은 걸 int()형으로 변화하여 정수로 저장.
        print()
        # 출력 후 줄바꿈 start.
        if command == 0 and tree.root is not None:
            print(level_order(tree))
        elif command == 1:
            data = get_data("트리에 추가할 정수를 입력하세요.")
            if data is not None:
                tree.insert(data)
                print(f"{data}(을)를 트리에 추가했습니다.")
            else:
                print("값을 잘못 입력했습니다.")
        elif command == 2:
            data = get_data("트리에서 삭제할 정수를 입력하세요.")
            if data is not None:
                tree.delete(data)
                print(f"{data}(을)를 트리에서 삭제했습니다.")
            else:
                print("값을 잘못 입력했습니다.")
        elif command == 3:
            data = get_data("검색할 값을 입력하세요.")
            if data is not None:
                if data in tree:
                    print(f"{data}(이)가 리스트에 있습니다.")
                else:
                    print(f"{data}(이)가 리스트에 없습니다.")
            else:
                print("값을 잘못 입력했습니다.")
        elif command == 9:
            break

# Co - Occurence Network

class Occurnetwork:
    def __init__(self, *args):
        if not args:
            raise ValueError("error occured")
        self.nodes = list(args)
        self.edges = {} #딕셔너리 use.(초기화) = edge 표현하기 편하겟군

    def Makeedge(self, value):
        #오류 있을 때를 정의해놓아야지..
        if value not in range(len(self.nodes)):
            raise ValueError("ERROR")
        
        if value not in self.edges:
            self.edges[value] = []
import numpy
class lossfunction():
    def __init__(self, name, args):
        self.name = name
        self.lists = []
        self.lists.append(args)
        return len(self.lists)
    def connenction(self):
        loss = 1e-34
                
        
import requests
res = requests.get("")
res.raise_for_status()
print("code :", res.status_code)
