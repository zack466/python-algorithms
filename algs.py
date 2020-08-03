# Algorithms based on "Princeton Algorithms, Part 1" course
# Implemented in Python
# Zachary Huang
#
# %%
import random
import numpy as np
import math

# %%
class QuickFind:
    #represented by array
    #items are in the same connected component if their array value is the same
    def __init__(self, size):
        self.size = size
        self.id = list(range(size))

    def show(self):
        print(list(range(self.size)))
        print(self.id)

    def union(self, a, b):
        val1 = self.id[a]
        val2 = self.id[b]
        if val1==val2:
            return
        for i in range(self.size):
            if self.id[i] == val1:
                self.id[i] = val2

    def connected(self, a, b):
        return self.id[a] == self.id[b]

    @staticmethod
    def test():
        QF = QuickFind(10)
        QF.union(4,3)
        QF.union(3,8)
        QF.union(9,4)
        QF.union(2,1)

        QF.show()

#QuickFind.test()

# %%
class QuickUnion:
    #represented by array
    #id[i] points to the parent of i
    #items are connected if they have the same root
    def __init__(self, size, weighted=False, pathCompress=False):
        self.size = size
        self.id = list(range(size))
        if weighted:
            self.sz = [1] * self.size # size array for weighted union
        self.weighted = weighted
        self.pathCompress = pathCompress

    def show(self):
        print(list(range(self.size)))
        print(self.id)

    def root(self, a):
        while a != self.id[a]:
            if self.pathCompress:
                self.id[a] = self.id[self.id[a]] #basic 'grandparent' path compression
            a = self.id[a]
        return a

    def union(self, a, b):
        if self.weighted:
            self.weightedUnion(a,b)
        else:
            self.id[self.root(a)] = self.root(b)

    def weightedUnion(self, a, b):
        r1 = self.root(a)
        r2 = self.root(b)
        if (r1 == r2):
            return
        if self.sz[r1] < self.sz[r2]:
            self.id[r1] = r2
            self.sz[r2] += self.sz[r1]
        else:
            self.id[r2] = r1
            self.sz[r1] += self.sz[r2]

    def connected(self, a, b):
        return self.root(a) == self.root(b)

    def avgDepth(self):
        total = 0
        for i in range(self.size):
            counter = 0
            a = i
            while a != self.id[a]:
                a = self.id[a]
                counter += 1
            total += counter
        return total / self.size

    @staticmethod
    def test():
        QU = QuickUnion(10)
        betterQU = QuickUnion(10,weighted=True,pathCompress=True)

        QU.union(4,3)
        QU.union(3,8)
        QU.union(6,5)
        QU.union(9,4)
        QU.union(2,1)

        betterQU.union(4,3)
        betterQU.union(3,8)
        betterQU.union(6,5)
        betterQU.union(9,4)
        betterQU.union(2,1)

        QU.show()
        print(QU.avgDepth())
        print()
        betterQU.show()
        print(betterQU.avgDepth())

#QuickUnion.test()

# %%
class Percolation:
    # grid looks like (for n=3):
    # 0  <-- virtual top site
    # 1 2 3
    # 4 5 6
    # 7 8 9
    # 10 <-- virtual bottom site
    def __init__(self, n):
        self.n = n
        self.grid = QuickUnion(n**2 + 2, weighted=True) #extra two virtual sites at top and bottom
        self.opened = [False] * (n**2 + 2)
        for i in range(n):
            self.grid.union(0,i+1)
            self.grid.union(n**2 + 1,n**2 - i)

    def surroundings(self, a):
        sur = []
        # get above
        if a > self.n:
            sur.append(a-self.n)
        # get below
        if a <= self.n**2 - self.n:
            sur.append(a+self.n)
        # get left
        if a % self.n != 1:
            sur.append(a - 1)
        # get right
        if a % self.n != 0:
            sur.append(a+1)

        return sur

    def open(self, a):
        self.opened[a] = True
        for node in self.surroundings(a):
            if self.opened[node]:
                self.grid.union(a, node)

    def show(self):
        #self.grid.show()
        for i in range(self.n):
            copy = self.grid.id[1:-1]
            print(copy[i*self.n:(i+1)*self.n])

    def percolates(self):
        return self.grid.connected(0,self.n**2 + 1)

    @staticmethod
    def test():
        n = 16
        results = []
        for i in range(1000):
            counter = 0
            p = Percolation(n)
            all = list(range(1,n**2 + 1))
            while not p.percolates():
                r = random.choice(all)
                all.remove(r)
                p.open(r)
                counter += 1
            results.append(counter / (n**2))
        print("Average percolation threshold: ")
        print(sum(results)/len(results))

#Percolation.test()

# %%
class Node:
    # a linked list node
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        next = self.next if self.next is None else self.next.val
        return "Val: " + str(self.val) + ", next: " + str(next)

class DNode:
    def __init__(self, val=None, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev

    def __str__(self):
        next = self.next if self.next is None else self.next.val
        prev = self.prev if self.prev is None else self.prev.val
        return "Val: " + str(self.val) + ", next: " + str(next) + ", prev: " + str(prev)

# %%
class Stack:
    #implemented with linked lists
    def __init__(self):
        self.first = None

    def __iter__(self):
        self.iter = self.first
        return self

    def __next__(self):
        if self.iter == None:
            raise StopIteration
        else:
            val = self.iter.val
            self.iter = self.iter.next
            return val

    def push(self, val):
        oldFirst = self.first
        self.first = Node()
        self.first.val = val
        self.first.next = oldFirst

    def pop(self):
        if self.isEmpty():
            return None
        val = self.first.val
        self.first = self.first.next
        return val

    def isEmpty(self):
        return self.first == None

    def show(self):
        cur = self.first
        while cur != None:
            print(cur.val)
            cur = cur.next

arrayStack = [] #stack implemented as array

# %%
class Queue:
    #implemented with linked lists
    def __init__(self):
        self.first = None
        self.last = None

    def __iter__(self):
        self.iter = self.first
        return self

    def __next__(self):
        if self.iter == None:
            raise StopIteration
        else:
            val = self.iter.val
            self.iter = self.iter.next
            return val

    def enqueue(self, val):
        last = self.last
        self.last = Node()
        self.last.val = val
        self.last.next = None
        if self.isEmpty():
            self.first = self.last
        else:
            last.next = self.last

    def dequeue(self):
        if self.isEmpty():
            return
        val = self.first.val
        self.first = self.first.next
        if self.isEmpty():
            self.last = None
        return val

    def isEmpty(self):
        return self.first == None

    def show(self):
        cur = self.first
        while cur != None:
            print(cur.val)
            cur = cur.next

class Deque:
    def __init__(self):
        self.first = None
        self.last = None

    def insertFront(self, val):
        first = self.first
        self.first = DNode()
        self.first.val = val
        self.first.prev = first
        self.first.next = None
        if first == None:
            self.last = self.first
        else:
            first.next = self.first

    def insertRear(self, val):
        last = self.last
        self.last = DNode()
        self.last.val = val
        self.last.prev = None
        self.last.next = last
        if last == None:
            self.first = self.last
        else:
            last.prev = self.last

    def deleteFront(self):
        if self.isEmpty():
            return
        first = self.first.prev
        self.first = first
        if self.first != None:
            self.first.next = None
        else:
            self.last = None

    def deleteRear(self):
        if self.isEmpty():
            return
        last = self.last.next
        self.last = last
        if self.last != None:
            self.last.prev = None
        else:
            self.first = None

    def show(self):
        last = self.last
        #print("last: " + str(self.last))
        #print("first: " + str(self.first))
        while last != None:
            print(last.val)
            last = last.next

    def isEmpty(self):
        return self.first == None and self.last == None

    @staticmethod
    def test():
        a = Deque()

        a.insertFront(10)
        a.insertRear(3)
        a.insertFront(11)
        a.insertRear(44)

        a.deleteFront()
        a.deleteRear()

        a.show()

# %%
from ast import literal_eval

def isInt(s):
    val = literal_eval(s)
    return isinstance(val, int) or (isinstance(val, float) and val.is_integer()) #accounts for 0s after decimal place

def isFloat(s):
    val = literal_eval(s)
    return isinstance(val, float) and not val.is_integer()

class TwoStackAlgorithm:
    # Dijkstra's two-stack algorithm for expression evaluation
    class TokenError(Exception):
        def __init__(self, token):
            super().__init__("Token not recognized: " + token)

    def __init__(self, expression):
        self.expression = expression
        self.valStack = Stack()
        self.opStack = Stack()
        self.charToOp = {   "+":lambda a,b: a+b,
                            "-":lambda a,b: a-b,
                            "*":lambda a,b: a*b,
                            "/":lambda a,b: a/b,
                            "^":lambda a,b: a**b}
        self.ops = self.charToOp.keys()

    def evaluate(self):
        toks = self.expression.split()
        for i in range(len(toks)):
            tok = toks[i]
            if tok == "(": #ignore left parentheses
                pass
            elif tok in self.ops: #push operator onto op stack
                self.opStack.push(self.charToOp[tok])
            elif tok == ")": # calculate result from top two values and top operator, then push onto val stack
                val2, val1 = self.valStack.pop(), self.valStack.pop() #values are backwards on stack
                operator = self.opStack.pop()
                result = operator(val1,val2)
                self.valStack.push(result)
            elif isInt(tok):
                self.valStack.push(int(tok))
            elif isFloat(tok):
                self.valStack.push(float(tok))
            else:
                raise self.TokenError(tok) #not recognized
        return self.valStack.pop()  #final result is last value on value stack

    @staticmethod
    def test():
        a = TwoStackAlgorithm("( ( 6.2 ^ 2 ) + ( ( 1 / 2 ) * 3 ) )")
        print(a.evaluate())

#TwoStackAlgorithm.test()

# %%
class Shuffle:
    @staticmethod
    def randomShuffle(arr):
        #assigns each element a random value and then sorts the array
        rand = [{"rand":random.random(),"val":arr[i]} for i in range(len(arr))]
        rand.sort(key=lambda x: x["rand"])
        arr[:] = [x["val"] for x in rand]

    @staticmethod
    def knuthShuffle(arr):
        #uniform random shuffling in linear time
        #like insertion sort but with random swapping instead
        for i in range(len(arr)):
            rand = random.choice(list(range(0,i+1)))
            arr[i], arr[rand] = arr[rand], arr[i]

    @staticmethod
    def test():
        ls = [1,2,3,4,5,6,7,8,9,10]
        print(ls)
        #Shuffle.randomShuffle(ls)
        Shuffle.knuthShuffle(ls)
        print(ls)

#Shuffle.test()

# %%
class SelectionSort:
    @staticmethod
    def sort(arr):
        for i in range(len(arr)):
            minidx = np.argmin(arr[i:]) + i #add i to compensate for shorter list
            arr[i],arr[minidx] = arr[minidx],arr[i]

    @staticmethod
    def test():
        ls = [1,5,5,9,7,6,14,2]
        print(ls)
        SelectionSort.sort(ls)
        print(ls)

#Selection.test()

# %%
class InsertionSort:
    @staticmethod
    def sort(arr):
        for i in range(len(arr)):
            j = i
            while arr[j] < arr[j-1] and j > 0:
                arr[j-1], arr[j] = arr[j], arr[j-1]
                j -= 1

    @staticmethod
    def test():
        ls = [1,5,10,9,7,6,14,2]
        print(ls)
        InsertionSort.sort(ls)
        print(ls)

#Insertion.test()

# %%
class ShellSort:
    @staticmethod
    def hsort(arr, h):
        #insertion sort but with stride h
        for i in range(h, len(arr)):
            j = i
            while j>=h and arr[j] < arr[j-h]:
                arr[j], arr[j-h] = arr[j-h], arr[j]
                j -= h

    @staticmethod
    def sort(arr):
        incr = [1]
        while incr[-1] < len(arr):
            incr.append(incr[-1]*3 + 1)
        for h in incr[::-1]: #increments of form 3x+1
            ShellSort.hsort(arr,h)

    @staticmethod
    def test():
        ls = [3,5,10,9,7,6,14,2,6,1,11,17,4]
        print(ls)
        ShellSort.sort(ls)
        print(ls)

#Shell.test()

# %%
class MergeSort:
    @staticmethod
    def merge(arr, low, mid, high):
        copy = [] #possibly poor performance? better to create copy array outside of merge/sort functions
        for e in arr: #need deep copy
            copy.append(e)
        i = low
        j = mid + 1
        for k in range(low, high+1):
            if i > mid:
                arr[k] = copy[j]
                j += 1
            elif j > high:
                arr[k] = copy[i]
                i += 1
            elif copy[j] < copy[i]:
                arr[k] = copy[j]
                j += 1
            else:
                arr[k] = copy[i]
                i+= 1

    @staticmethod
    def bottomUpSort(arr):
        n = len(arr)
        #copy = []
        sz = 1
        while sz < n:
            low = 0
            while low < n - sz:
                MergeSort.merge(arr, low, low+sz-1, min(low+2*sz-1, n-1))
                low += 2*sz
            sz *= 2

    @staticmethod
    def sort(arr, low=None, high=None, nonRecursive=False):
        if nonRecursive:
            MergeSort.bottomUpSort(arr)
            return
        if low == None:
            low = 0
        if high == None:
            high = len(arr)-1
        if high <= low:
            return
        mid = low + (high - low) // 2
        MergeSort.sort(arr, low=low, high=mid)
        MergeSort.sort(arr, low=mid+1, high=high)
        MergeSort.merge(arr, low, mid, high)

    @staticmethod
    def test():
        ls = [3,5,11,9,7,6,14,2,6,1,11,17,4]
        print(ls)
        MergeSort.sort(ls, nonRecursive=True)
        print(ls)

#MergeSort.test()

# %%
class QuickSort:
    """
    -Many variants, two main textbook partitioning systems (hoare, lomuto)
    -Recursive divide-and-conquer sorting algorithm
    -Steps:
        1. Choose a pivot and find its true value
        2. Partition the array into two arrays using the pivot's position
        3. Repeat but with each smaller array
    """
    #CLRS implementation -- Lomuto QuickSort
    @staticmethod
    def lomuto(ls,lo=None,hi=None):
        if lo==None and hi==None:
            lo = 0
            hi = len(ls) - 1
            Shuffle.randomShuffle(ls)
        if lo<hi:
            pi = QuickSort.lomuto_partition(ls,lo,hi)
            QuickSort.lomuto(ls,lo,pi-1)
            QuickSort.lomuto(ls,pi+1,hi)

    @staticmethod
    def lomuto_partition(ls,lo,hi):
        """
        -Loop invariant of 4 sections: [vals < pivot][vals > pivot][unsorted][pivot]
        -Afterwards, moves pivot in between first two sections
        """
        pivot = ls[hi]
        i = lo-1
        for j in range(lo,hi):
            if ls[j] <= pivot:
                i += 1
                ls[i],ls[j] = ls[j],ls[i]
        ls[i+1],ls[hi] = ls[hi],ls[i+1]
        return i+1

    @staticmethod
    def hoare(ls,lo=None,hi=None):
        if lo==None and hi==None:
            lo = 0
            hi = len(ls) - 1
            Shuffle.randomShuffle(ls)
        if lo<hi:
            pi = QuickSort.hoare_partition(ls,lo,hi)
            QuickSort.hoare(ls,lo,pi)
            QuickSort.hoare(ls,pi+1,hi)

    @staticmethod
    def hoare_partition(ls,lo,hi):
        """
        -Has a left/right pointer that moves towards true position of pivot and swaps values that are out of order
        -Afterwards, the pivot is in its true position
        """
        pivot = ls[lo]
        i = lo-1
        j = hi + 1
        while True:
            while True:
                j -= 1
                if ls[j] > pivot:
                    pass
                else:
                    break
            while True:
                i += 1
                if ls[i] < pivot:
                    pass
                else:
                    break
            if i < j:
                ls[i],ls[j] = ls[j],ls[i]
            else:
                break
        return j

    @staticmethod
    def threeWaySort(arr, low=None, high=None):
        #much faster when many keys are equal, uses 3-way partitioning
        if low == None and high == None:
            low = 0
            high = len(arr) - 1
            Shuffle.randomShuffle(ls)
        if high <= low:
            return
        lt = low
        gt = high
        pivot = arr[low]
        i = low
        while i <= gt:
            if arr[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i += 1
            elif arr[i] > pivot:
                arr[i], arr[gt] = arr[gt], arr[i]
                gt -= 1
            else:
                i += 1
        QuickSort.threeWaySort(arr, low, lt-1)
        QuickSort.threeWaySort(arr, gt+1, high)

    @staticmethod
    def partition(arr, low, high):
        i = low + 1
        j = high
        while True:
            while arr[i] < arr[low]:
                i += 1
                if i == high:
                    break
            while arr[low] < arr[j]:
                j -= 1
                if j == low:
                    break
            if (i >= j):
                break
            arr[i], arr[j] = arr[j], arr[i]
        arr[low], arr[j] = arr[j], arr[low]
        return j

    @staticmethod
    def sort(ls, low=None, high=None):
        if low==None and high==None: #if calling for first time
            low = 0
            high = len(ls) - 1
            Shuffle.randomShuffle(ls) # shuffle for performance guarantee
        if high <= low:
            return
        j = QuickSort.partition(ls, low, high)
        QuickSort.sort(ls, low, j-1)
        QuickSort.sort(ls, j+1, high)

    @staticmethod
    def test():
        ls = [3,5,11,9,7,6,14,2,6,1,11,17,4]
        print(ls)
        #QuickSort.sort(ls)
        #QuickSort.hoare(ls)
        #QuickSort.lomuto(ls)
        QuickSort.threeWaySort(ls)
        print(ls)

#QuickSort.test()

# %%
def is_sorted(ls):
    """
    -Checks if a given list is sorted
    """
    for i in range(len(ls)-1):
        if ls[i] > ls[i+1]:
            return False
    return True

class BubbleSort:
    """
    -"In short, the bubble sort seems to have nothing to recommend it, except a catchy name and the fact that it leads to some interesting theoretical problems."
        ~ D.E. Knuth, The Art of Computer Programming (1973)
    """
    @staticmethod
    def sort(ls):
        """
        -Compares each two consecutive elements of an array and compares them, swapping them if they are out of order
        -In turn, large numbers "float" towards to the top of the array ==> "bubble" sort
        """
        while not is_sorted(ls):
            for i in range(0,len(ls)-1):
                if ls[i] > ls[i+1]:
                    ls[i],ls[i+1] =  ls[i+1],ls[i]
    @staticmethod
    def test():
        ls = [3,5,11,9,7,6,14,2,6,1,11,17,4]
        print(ls)
        BubbleSort.sort(ls)
        print(ls)

#BubbleSort.test()

class CocktailSort:
    @staticmethod
    def sort(ls):
        """
        -Aka "bidirectional bubble sort" aka "shaker sort" aka "cocktail sort" aka "shuttle sort"
        -Essentially just bubble but going up and down repeatedly
        -Is faster than vanilla bubble sort
        """
        while not is_sorted(ls):
            for i in range(0,len(ls)-1):
                if ls[i] > ls[i+1]:
                    ls[i],ls[i+1] =  ls[i+1],ls[i]
            for i in range(len(ls)-1,0,-1):
                if ls[i] < ls[i-1]:
                    ls[i],ls[i-1] =  ls[i-1],ls[i]
    @staticmethod
    def test():
        ls = [3,5,11,9,7,6,14,2,6,1,11,17,4]
        print(ls)
        CocktailSort.sort(ls)
        print(ls)

#CocktailSort.test()

class BogoSort:
    """
    -Randomizes list until it's sorted
    -Average time complxity of O(n*n!); Worst case is O(∞) <-- lmao
    """
    @staticmethod
    def sort(ls):
        while not is_sorted(ls):
            Shuffle.knuthShuffle(ls)

    ### DO NOT RUN ###
    @staticmethod
    def test():
        ls = [3,5,11,9,7,6,14,2,6,1,11,17,4]
        print(ls)
        BogoSort.sort(ls)
        print(ls)

#BogoSort.test()

# %%
class TopK:
    @staticmethod
    def quickSelect(arr, k):
        Shuffle.randomShuffle(arr)
        low = 0
        high = len(arr) - 1
        while high > low:
            j = QuickSort.partition(arr, low, high)
            if j < k:
                low = j + 1
            elif j > k:
                high = j - 1
            else:
                return arr[k]
        return arr[k]

    @staticmethod
    def test():
        ls = [3,5,11,9,7,6,14,2,6,1,11,17,4]
        print("unsorted: " + str(ls))
        median = TopK.quickSelect(ls, len(ls)//2)
        QuickSort.sort(ls)
        print("quickselect: " + str(median))
        print("actual: " + str(ls[len(ls)//2]))

#TopK.test()

# %%
class UnorderedPriorityQueue:
    #implemented with python standard array
    def __init__(self):
        self.n = 0
        self.ls = []

    def insert(self, val):
        self.ls.append(val)
        self.n += 1

    def deleteMax(self):
        if self.isEmpty():
            return
        max = 0 #max index
        for i in range(self.n):
            if self.ls[i] > self.ls[max]:
                max = i
        self.ls[max], self.ls[self.n-1] = self.ls[self.n-1], self.ls[max]
        self.n -= 1
        return self.ls.pop()

    def isEmpty(self):
        return self.n==0

    def show(self):
        print(self.ls)

    @staticmethod
    def test():
        a = UnorderedPriorityQueue()
        a.insert(10)
        a.insert(14)
        a.insert(2)
        a.deleteMax()
        a.show()

#UnorderedPriorityQueue.test()

class OrderedPriorityQueue:
    def __init__(self):
        self.n = 0
        self.ls = []

    def insert(self, val):
        self.ls.append(val)
        self.n += 1
        #simplified insertion sort alg to insert new value
        j = self.n - 1
        while self.ls[j] < self.ls[j - 1] and j>0:
            self.ls[j-1], self.ls[j] = self.ls[j], self.ls[j-1]
            j -= 1

    def deleteMax(self):
        if self.isEmpty():
            return
        return self.ls.pop()

    def isEmpty(self):
        return self.n==0

    def show(self):
        print(self.ls)

    @staticmethod
    def test():
        a = OrderedPriorityQueue()
        a.insert(10)
        a.insert(14)
        a.insert(2)
        a.deleteMax()
        a.show()

#OrderedPriorityQueue.test()

class BinaryHeap:
    # heap ordering: parent val greater or equal than children val
    # indices start at 1, nodes written in level order (like BFS)
    # array arithmetic: largest node at arr[1]
    #                   parent of k at k // 2
    #                   children of k at 2k and 2k+1
    # promotion: if heap order violated, exchange child with parent until order is restored
    def __init__(self):
        self.ls = [None]
        self.n = 0

    def swim(self, k):
        #promote node at k
        while (k>1 and self.ls[k//2] < self.ls[k]):
            self.ls[k//2], self.ls[k] = self.ls[k], self.ls[k//2]
            k = k//2

    def insert(self, val):
        #add a node, swim it up
        self.ls.append(val)
        self.n += 1
        self.swim(self.n)

    def sink(self, k):
        #demote node at k until order is restored
        while (2*k <= self.n):
            j = 2*k
            if j < self.n and (self.ls[j] < self.ls[j+1]):
                j += 1
            if self.ls[k] >= self.ls[j]:
                break
            self.ls[k], self.ls[j] = self.ls[j], self.ls[k]
            k = j

    def deleteMax(self):
        # replace root with last node, pop last node, and sink the new root down
        if self.isEmpty():
            return
        max = self.ls[1]
        self.ls[1], self.ls[self.n] = self.ls[self.n], self.ls[1]
        self.ls.pop()
        self.n -= 1
        self.sink(1)
        return max

    def show(self):
        print(self.ls)

    def isEmpty(self):
        return self.n==0

    @staticmethod
    def test():
        a = BinaryHeap()
        a.insert(2)
        a.insert(1)
        a.insert(16)
        a.insert(4)
        a.insert(10)
        a.show()
        a.deleteMax()
        a.show()

#BinaryHeap.test()

class HeapSort:
    @staticmethod
    def sort(arr):
        arr.insert(0,None) #insert placeholder for 1 based indexing
        n = len(arr) - 1
        #get into heap order
        for k in range(n//2, 0, -1):
            HeapSort.sink(arr, k, n)
        #put max at end of array repeatedly
        while n>1:
            arr[1], arr[n] = arr[n], arr[1]
            n -= 1
            HeapSort.sink(arr, 1, n)
        #remove placeholder at beginning
        arr.pop(0)

    @staticmethod
    def sink(arr, k, n): #copied from binary heap, slightly edited
        while (2*k <= n):
            j = 2*k
            if j < n and (arr[j] < arr[j+1]):
                j += 1
            if arr[k] >= arr[j]:
                break
            arr[k], arr[j] = arr[j], arr[k]
            k = j

    @staticmethod
    def test():
        ls = [3,3,7,5,11,9,7,6,14,2,6,1,11,17,4]
        print(ls)
        HeapSort.sort(ls)
        print(ls)

#HeapSort.test()

# %%
class STNode:
    # a symbol table linked list node
    def __init__(self, key=None, val=None, next=None):
        self.key = key
        self.val = val
        self.next = next

    def __str__(self):
        next = self.next if self.next is None else self.next.val
        return "Key: " + str(self.key) + ", Val: " + str(self.val)

class UnorderedListSymbolTable:
    def __init__(self):
        self.start = None

    def put(self, key, val):
        if self.isEmpty():
            self.start = STNode(key=key,val=val)
        else:
            scan = self.start
            while scan.key != key and scan.next != None:
                scan = scan.next
            if scan.key == key:
                scan.val = val
            else:
                scan.next = STNode(key=key,val=val)

    def get(self, key):
        scan = self.start
        while scan.key != key and scan.next != None:
            scan = scan.next
        if scan.key == key:
            return scan.val
        else:
            return None

    def delete(self, key): #fake delete
        self.put(key, None)

    def contains(self, key):
        return self.get(key) != None

    def isEmpty(self):
        return self.start == None

    def show(self):
        scan = self.start
        while scan != None:
            if self.contains(scan.key): #ignores "deleted" keys
                print(scan)
            scan = scan.next
    @staticmethod
    def test():
        a = LinkedListSymbolTable()

        a.put("a",1)
        a.put("b",2)
        a.put("c",10)
        a.put("b",10)
        a.delete("a")

        a.show()

class OrderedSymbolTable:
    #keeps keys in order, uses binary search to get correct key
    #ordered keys allow for convenient operations like min, max, floor, ceil, select, range, etc
    def __init__(self):
        self.keys = []
        self.vals = []
        self.n = 0

    def get(self, key):
        if self.isEmpty():
            return
        i = self.rank(key)
        if i < self.n and self.keys[i] == key:
            return self.vals[i]
        else:
            return

    def put(self, key, val):
        if key in self.keys:
            self.vals[self.rank(key)] = val
        else:
            self.keys.append(key)
            self.vals.append(val)
            j = self.n
            self.n += 1
            while self.keys[j] < self.keys[j-1] and j > 0:
                self.keys[j-1], self.keys[j] = self.keys[j], self.keys[j-1]
                self.vals[j-1], self.vals[j] = self.vals[j], self.vals[j-1]
                j -= 1

    def rank(self, key):
        #returns number of keys less than key
        low = 0
        high = self.n-1
        while low < high:
            mid = low + (high-low) // 2
            if key < self.keys[mid]:
                high = mid - 1
            elif key > self.keys[mid]:
                low = mid + 1
            else:
                return mid
        return low

    def delete(self, key):
        if key in self.keys:
            idx = self.rank(key)
            self.keys.pop(idx)
            self.vals.pop(idx)
            self.n -= 1

    def isEmpty(self):
        return self.n == 0

    def show(self):
        print(self.keys)
        print(self.vals)

    @staticmethod
    def test():
        a = OrderedSymbolTable()
        a.put(1, 4)
        a.put(4,10)
        a.put(3, 7)
        a.put(2, 6)
        a.put(3,2)
        a.delete(4)
        a.show()

#OrderedSymbolTable.test()

# %%
class BSTNode:
    #Binary search tree node
    def __init__(self, key=None, val=None, count=0):
        self.key = key
        self.val = val
        self.left = None
        self.right = None
        self.count = count

class BinarySearchTree:
    #a binary tree in symmetric order
    #each node key is larger than all keys in its left subtree and smaller than all keys in its right subtree
    #to search, go either left or right each node until you reach the correct key (or None)
    def __init__(self):
        self.root = None

    def __iter__(self):
        self.queue = Queue()
        self.inorder(self.root, self.queue)
        return iter(self.queue)

    def inorder(self, x, q):
        if x == None:
            return
        self.inorder(x.left, q)
        q.enqueue(x.key)
        self.inorder(x.right,q)

    def put(self, key, val):
        self.root = self.rec_put(self.root, key, val)

    def rec_put(self, x, key, val):
        #recursive put method to replace a value or create new node for a given key
        if x == None:
            return BSTNode(key, val, 1)
        if key < x.key:
            x.left = self.rec_put(x.left, key, val)
        elif key > x.key:
            x.right = self.rec_put(x.right, key, val)
        else:
            x.val = val
        x.count = 1 + self.rec_size(x.left) + self.rec_size(x.right)
        return x

    def get(self, key):
        x = self.root
        while (x != None):
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            else:
                return x.val
        return None

    def delete(self, key):
        pass

    def min(self):
        #returns smallest key in table
        x = self.root
        if self.isEmpty():
            return
        while x.left != None:
            x = x.left
        return x.key

    def max(self):
        #returns largest key in table
        x = self.root
        if self.isEmpty():
            return
        while x.right != None:
            x = x.right
        return x.key

    def floor(self, key):
        x = self.rec_floor(self.root, key)
        if x == None:
            return
        else:
            return x.key

    def rec_floor(self, x, key):
        if x == None:
            return None
        if x.key == key:
            return x
        if key < x.key:
            return self.rec_floor(x.left, key)
        t = self.rec_floor(x.right, key)
        if t != None:
            return t
        else:
            return x

    def isEmpty(self):
        return self.root == None

    def size(self):
        return self.rec_size(self.root)

    def rec_size(self, x):
        if x == None:
            return 0
        return x.count

    def rank(self, key):
        #gives number of keys less than a given key
        return self.rec_rank(key, self.root)

    def rec_rank(self, key, x):
        if x==None:
            return 0
        if key < x.key:
            return self.rec_rank(key, x.left)
        elif key > x.key:
            return 1 + self.rec_size(x.left) + self.rec_rank(key, x.right)
        else:
            return self.rec_size(x.left)

    def show(self, x=None):
        if x==None:
            return
        print(str(x.key) + " " + str(x.val))

        self.show(x.left)
        self.show(x.right)

    @staticmethod
    def test():
        a = BinarySearchTree()

        a.put(6,2)
        a.put(3,2)
        a.put(5,2)
        a.put(2,4)
        a.put(7,2)

        print(a.floor(4))
        print(a.size())
        print(a.rank(5))
        a.show(x=a.root)

        for key in a:
            print(key)

#BinarySearchTree.test()

# %%
class HashTable:
    # items are stored in slots generated from hashing the key
    def __init__(self, m):
        self.m = m
        pass

    def divisionHash(key):
        return k % self.m

    def multiplicationHash(key):
        A = (math.sqrt(5) - 1) / 2 #a constant between 0 and 1
        ka = key * A
        return int(self.m * (ka - int(ka)))
