# Algorithms based on "Princeton Algorithms, Part 1" course
# Implemented in Python
# Zachary Huang
#
# %%
import random
import numpy as np

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

# %%
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
            if tok == "(":
                pass
            elif tok.isnumeric():
                self.valStack.push(int(tok))
            elif tok in self.ops:
                self.opStack.push(self.charToOp[tok])
            elif tok == ")":
                val2, val1 = self.valStack.pop(), self.valStack.pop() #values are backwards on stack
                operator = self.opStack.pop()
                result = operator(val1,val2)
                self.valStack.push(result)
            else:
                raise self.TokenError(tok)
        return self.valStack.pop()

    @staticmethod
    def test():
        a = TwoStackAlgorithm("( ( 6 ^ 2 ) + ( ( 1 / 2 ) * 3 ) )") #final result is last value on value stack
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
    -Has two different partitioning algorithm - Lomuto and Hoare - but both are called "QuickSort"
    (the actual recursive algorithm is still the same)
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
        QuickSort.sort(ls)
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
