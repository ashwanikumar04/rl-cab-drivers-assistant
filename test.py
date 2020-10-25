m = 5
#print([ (i,j) for i in range(m) for j in range(m) if i!=j])

l = [(0, 0)]+[(i, j) for i in range(m) for j in range(m) if i != j]
#l=[(0,0)]+[(p,q) for p in range(m) for q in range(m) if p!=q]
# print(*l, sep="\n")
import random


print(23//24)

t = 24
d = 7

action_space = [(0, 0)]+[(p, q) for p in range(m)
                                for q in range(m) if p != q]
state_space = [(xi, tj, dk) for xi in range(m)
                    for tj in range(t) for dk in range(d)]


print(len(action_space))


