import numpy as np
import itertools

a = [1,]
print(a[:])

# for i in range(100):
#     print(np.random.choice(a=[0,1], p=[0.3,0.7]))

a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([2,4,6,2,4,6,2,4,6])
# print(b[-1:])
# print(np.square(a - b))
# print(sum(np.square(a - b)))
# print(np.sqrt(sum(np.square(a - b))))

# for s, s_ in zip(a[:-1], a[1:]):
#     print((s,s_))

num_state = 7
print([i/(num_state-1) for i in range(num_state)])

# c = [1,2,3]
# d = [4,5,6]
#
# for x in itertools.product(a,b):
#     print(x)

print(a[-3:])
print(a[-3:-1])
print(a[:-1])
print(a[-1:])

print([i for i in range(1,10)])
print([i for i in range(10)])
