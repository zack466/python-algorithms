import sort
from sort import Quicksort
from sort import Bubblesort
from sort import is_sorted

alist = []
blist = []
print("started")

ls = sort.rand_list(20)
print(ls)
print("Is sorted: " + str(is_sorted(ls)))

#Quicksort.sort(ls,"Lumoto")
#Quicksort.sort(ls,"Hoare")
#Quicksort.sort(ls,"Zack")
#Bubblesort.sort(ls)
#Bubblesort.shaker_sort(ls)
#sort.Bogosort.sort(ls)

print(ls)
print("Is sorted: " + str(is_sorted(ls)))
