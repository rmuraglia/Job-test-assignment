
# fix 1: needed to change %i for name to %s for printing strings
friends = ['john', 'pat', 'gary', 'michael']
for i, name in enumerate(friends):
    print "name %i is %s" % (i, name)

# How many friends contain the letter 'a' ?
# fix 2: search for 'a' as string (not variable a) in name
# improper syntax for incrementing count_a
# force float in division to do non-integer division
count_a = 0
for name in friends:
    if 'a' in name:
        count_a += 1 # or count_a = count_a + 1

print "%f percent of the names contain an 'a'" % ( count_a / float(len(friends)) * 100 )


# Say hi to all friends
# fix 3: change order of arguments in function definition. Arguments with defaults should be last
def print_hi(name, greeting='hello') :
    print "%s %s" % (greeting, name)

map(print_hi, friends)

# Print sorted names out
# fix 4: the sort command just sorts in place and doesn't return anything. if you want to print it ou, you have to call print on the list itself after sorting
friends.sort()
print friends

# fix 5: improper block comment. Use triple quotes or pound sign instead

#    Calculate the factorial N! = N * (N-1) * (N-2) * ...

# fix 6: correct if/else statement syntax, correct recursive calculation of factorial
# note: docstring seems ambiguous. I would write :param x: instead of :param N:, but it doesn't affect the code's performance
def factorial(x):
    """
    Calculate factorial of number
    :param N: Number to use
    :return: x!
    """
    if x==1: return 1
    else : return factorial(x-1) * x

print "The value of 5! is", factorial(5)
