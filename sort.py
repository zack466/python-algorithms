import random

def check_mountain(ls,idx):
	"""
	-Helper function for quicksort
	-Checks if number at index is in its true position
	"""
	for i in range(idx):
		if ls[i] > ls[idx]:
			return False
	for i in range(idx+1,len(ls)):
		if ls[i] < ls[idx]:
			return False
	return True

def is_sorted(ls):
	"""
	-Checks if a given list is sorted
	"""
	for i in range(len(ls)-1):
		if ls[i] > ls[i+1]:
			return False
	return True

def rand_list(n,lower=0,upper=100):
	"""Generates a random list of integers with n terms and specified lower/upper bounds"""
	ls = []
	for i in range(n):
		ls.append(random.randrange(lower,upper))
	return ls

class Quicksort:
	"""
	-Has two different partitioning algorithm - Lomuto and Hoare - but both are called "quicksort"
	(the actual recursive algorithm is still the same)
	-Recursive divide-and-conquer sorting algorithm
	-Steps:
		1. Choose a pivot and find its true value
		2. Partition the array into two arrays using the pivot's position
		3. Repeat but with each smaller array
	"""
	#CLRS implementation -- Lomuto quicksort
	def lomuto(ls,lo,hi):
		if lo<hi:
			pi = Quicksort.lomuto_partition(ls,lo,hi)
			Quicksort.lomuto(ls,lo,pi-1)
			Quicksort.lomuto(ls,pi+1,hi)
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
	#Zack-Hoare partition (fake)
	def Zack(ls,lo,hi):
		if lo<hi:
			pi = Quicksort.Zack_partition(ls,lo,hi)
			Quicksort.Zack(ls,lo,pi-1)
			Quicksort.Zack(ls,pi+1,hi)
	def Zack_partition(ls,lo,hi):
		"""
		Hoare quicksort but pivot is only swapped once (at the end)
		"""
		#pivot is first index
		pivot = ls[lo]
		i = lo
		j = hi+1
		while True:
			while True: #move hi pointer left until less than pivot
				j -= 1
				if ls[j] > pivot and i!=j:
					pass
				else:
					break
			while True:
				i += 1
				if ls[i] < pivot and i!=j:
					pass
				else:
					break
			if i < j:
				ls[i],ls[j] = ls[j],ls[i]
			else:
				ls[lo],ls[j] = ls[j],ls[lo]
				return j
	def hoare(ls,lo,hi):
		if lo<hi:
			pi = Quicksort.hoare_partition(ls,lo,hi)
			Quicksort.hoare(ls,lo,pi)
			Quicksort.hoare(ls,pi+1,hi)
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
	def sort(ls,sort_type="Lomuto"):
		"""
		Sort types: Lomuto, Zack, Hoare
		"""
		n=len(ls)
		if sort_type=="Lomuto":
			Quicksort.lomuto(ls,0,n-1)
		elif sort_type=="Zack":
			Quicksort.Zack(ls,0,n-1)
		elif sort_type=="Hoare":
			Quicksort.hoare(ls,0,n-1)
		else:
			print("Sort type not found.")

class Bubblesort:
	"""
	-"In short, the bubble sort seems to have nothing to recommend it, except a catchy name and the fact that it leads to some interesting theoretical problems."
		~ D.E. Knuth, The Art of Computer Programming (1973)
	"""
	def sort(ls):
		"""
		-Compares each two consecutive elements of an array and compares them, swapping them if they are out of order
		-In turn, large numbers "float" towards to the top of the array ==> "bubble" sort
		"""
		comparisons = 0
		while not is_sorted(ls):
			for i in range(0,len(ls)-1):
				if ls[i] > ls[i+1]:
					ls[i],ls[i+1] =  ls[i+1],ls[i]
				else:
					pass
				comparisons += 1
		return comparisons
	def cocktail_sort(ls):
		"""
		-Aka "bidirectional bubble sort" aka "shaker sort" aka "cocktail sort" aka "shuttle sort"
		-Essentially just bubble but going up and down repeatedly
		-Is faster than vanilla bubble sort
		"""
		comparisons = 0
		while not is_sorted(ls):
			for i in range(0,len(ls)-1):
				if ls[i] > ls[i+1]:
					ls[i],ls[i+1] =  ls[i+1],ls[i]
				else:
					pass
				comparisons += 1
			for i in range(len(ls)-1,0,-1):
				if ls[i] < ls[i-1]:
					ls[i],ls[i-1] =  ls[i-1],ls[i]
				else:
					pass
				comparisons += 1
		return comparisons

class Bogosort:
	"""
	-Randomizes list until it's sorted
	-Average time complxity of O(n*n!); Worst case is O(∞) <-- lmao
	"""
	def sort(ls):
		while not is_sorted(ls):
			random.shuffle(ls)