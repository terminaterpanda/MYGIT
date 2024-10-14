# 1. BFS

# 1-1. BFS [with no collections package]

from random import shuffle


class Node:
    def __init__(self, data):
        if not data:
            raise ValueError("nothing is imported")
        self.data = data
        self.neighbors = []  # 다중연결 == "list"


class MakeNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("nothing is imported")
        self.lists = {}

        for i, value in enumerate(args, start=1):
            current = Node(value)  # 전달 value값을 이용해서 node 객체 생성
            self.lists[i] = current  # node를 dictionary self.lists에 저장

    def connect(self, key1, key2):
        node1 = self.lists[key1]
        node2 = self.lists[key2]

        node1.neighbors.append(node2)
        node2.neighbors.append(node1)  # 양방향 연결

    def bfs(self, start_key):
        if start_key not in self.lists:
            raise ValueError("error")

        start_node = self.lists[start_key]
        visited = set()  # visted를 set(집합) 형식으로 만들어서 value 값 부여.
        queue = [start_node]

        while queue:  # queue가 비어있지 않은 한 끊임없이 repeat.
            current_node = queue.pop(0)

            if current_node not in visited:
                print(f"node: {current_node.data}")
                visited.add(current_node)
                # node를 visited <set(집합)> 에 추가하여 중복 방지.

                for neighbor in current_node.neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)

        if not queue:
            print("visited everythng")
            queue.clear()


# 1-2. BFS [collections package]

from collections import deque


def bfs(graph, node, visited):
    queue = deque([node])
    visited[node] = True

    while queue:
        v = queue.popleft()
        print(v, end=" ")
        # 처리 중 node에서 방문하지 않은 node(인접) 전부 queue에 삽입
        for i in graph[v]:
            if not (visited[i]):
                queue.append(i)
                visited[i] = True


graph = [[], [2, 3], [1, 8], [1, 5], [], [3], [], [], [2]]

visited = [False] * 9

# dequeue == (양쪽 끝에서 삽입과 삭제를 지원)-> 더 빠른 속도(시간 복잡도)로 수행.


# 2. DFS


class Node:
    def __init__(self, data):
        if not data:
            raise ValueError("nothing is imported")
        self.data = data
        self.neighbors = []


class MakeNode:
    def __init__(self, *args):
        if not args:
            raise ValueError("error occured")
        self.lists = {}

        for i, value in enumerate(args, start=1):
            current = Node(value)
            self.lists[i] = current

    def connect(self, key1, key2):
        node1 = self.lists[key1]
        node2 = self.lists[key2]

        node1.neighbors.append(node2)
        node2.neighbors.append(node1)

    def dfs(self, start_key):
        if start_key not in self.lists:
            raise ValueError("error occured")

        visited = set()

        def dfs_gogo(node):
            if node in visited:
                return
            print(node.data, end=" ")
            visited.add(node)

            for neighbor in node.neighbors:
                dfs_gogo(neighbor)


# 2-1. DFS

from datamodeling import Stacker


class Node1:
    def __init__(self, data):
        if not data:
            raise ValueError("nothing is imported")
        self.data = data
        self.neighbors = []


class MakeNode1:
    def __init__(self, *args):
        if not args:
            raise ValueError("error occured")
        self.lists = {}

        for i, value in enumerate(args, start=1):
            current = Node(value)
            self.lists[i] = current

    def connect(self, key1, key2):
        node1 = self.lists[key1]
        node2 = self.lists[key2]

        node1.neighbors.append(node2)
        node2.neighbors.append(node1)

    def dfs_stacker(self, start_key):
        if start_key not in self.lists:
            raise ValueError("error occured")

        visited = set()
        stack = Stacker("MyStack", self.lists[start_key])

        while stack.peek_top():  # 스택이 비어 있지 않은 동안 반복.
            current_node = stack.pop()

            if current_node not in visited:
                print(current_node.data, end="")
                visited.add(current_node)

            for neighbor in reversed(current_node.neighbors):
                if neighbor not in visited:
                    stack.pushing(neighbor)


# 정렬(sort)

# 1. 버블정렬(Bubble)

import random


class MakeList:
    def __init__(self, name, *args):
        if not args:
            raise ValueError("error occurred")
        shuffled_list = list(args)
        random.shuffle(shuffled_list)
        self.name = shuffled_list

    def Bubble(self):
        n = len(self.name)
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.name[j] > self.name[j + 1]:
                    self.name[j], self.name[j + 1] = self.name[j + 1], self.name[j]
        print("the sorted list:", self.name)


# 2. 선택정렬(Selection Sort)


class Makelist1:
    def __init__(self, name, *args):
        if not args:
            raise ValueError("error occurred")
        list2 = list(args)
        random.shuffle(list2)
        self.name = list2

    def Selection(self):
        n = len(self.name)
        for i in range(n - 1):  # 쭉 반복.(i = 0부터 쭉 n-1까지 지속.)

            min_index = i
            for j in range(i + 1, n):
                if self.name[j] < self.name[min_index]:
                    min_index = j
                    # j는 i+1부터 n까지 쭉 이어져나가는 것이고, indexing 요소로 이루어짐.
            if min_index != i:
                self.name[i], self.name[min_index] = self.name[min_index], self.name[i]

        print("sorted list: ", self.name)


# 3. 삽입정렬(Insert Sort)


class Makelist2:
    def __init__(self, name, *args):
        if not args:
            raise ValueError("error occurred")
        list3 = list(args)
        random.shuffle(list3)
        self.name = list3

    def insert(self):
        n = len(self.name)
        for i in range(1, n):  # list의 두번째 요소부터 각 요소를 삽입과정
            kk = self.name[i]  # kk = 2
            j = i - 1  # j == 1

            while j >= 0 and self.name[j] > kk:
                self.name[j + 1] = self.name[j]
                j -= 1

            self.name[j + 1] = kk

        return self.name


# 4. Quick 정렬


class MakeList3:
    def __init__(self, name, *args):
        if not args:
            raise ValueError("error occurred")
        list3 = list(args)
        random.shuffle(list3)
        self.name = list3

    def Quick(
        self, a=None
    ):  # a = None 설정 이유:(재귀호출을 하는 경우에는 self.name을 직접 수정x)
        if a is None:
            a = self.name

        if len(a) <= 1:
            return a

        b = a[len(a) // 2]
        left = [x for x in a if x < b]  # left list
        middle = [x for x in a if x == b]  # middle list
        right = [x for x in a if x > b]  # right list

        return (
            self.Quick(left) + middle + self.Quick(right)
        )  # list 3개의 합인 하나의 리스트가 리턴.

    def Quicksortprint(self):
        stored = self.Quick()
        print(f"list: {stored}")
