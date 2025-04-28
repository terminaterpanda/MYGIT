mylist = [1,2,3,4,5,6,7,8,9,10]

even = [i for i in mylist if i % 2 == 0]
even = [i**2 for i in mylist if i % 2 == 0]

#set_comprehension
set_even = {i**2 for i in mylist if i % 2 == 0}
set_even

#dict comprehension
dict_even = {i:i**2 for i in mylist if i % 2 ==0}

