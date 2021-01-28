# a = [1,2,3,4]
# b = [2,3,4,5]

# ab = []

# for i in range(0, len(a)):
#    ab.append(a[i]*b[i])

# print(ab)

#istead of up with numpy

# import numpy as np

# a = np.array([1,2,3,4])
# b = np.array([2,3,4,5])

# print(a*b)






#numpy arrays

# import numpy as np

# a = np.array([1,2,3,4,5])
# print(type(a))

# b = np.array([1.4,2.5,3.5,5.5], dtype = "float32")
# print(b)

# d = np.array([1.4,2,3,5])
# print(d)
# #creating array from start
# c = np.zeros(10, dtype = int)
# print(c)

# e = np.ones((3,5), dtype = int)
# print(e)

# f = np.full((3,5), 3)
# print(f)

# g =  np.arange(0,31,3)
# print(g)

# h = np.linspace(0,1,10)
# print(h)

# i = np.random.normal(10, 4, (3,4))
# print(i)

# j = np.random.randint(0,10,(3,3))
# print(j)








#numpy array properties
# import numpy as np

# a = np.random.randint(10, size = 10)
# print(a)
# print(a.ndim)
# print(a.shape)
# print(a.size)
# print(a.dtype)
# print('-----------------')
# b = np.random.randint(10, size = (4,4))
# print(b)
# print(b.ndim)
# print(b.shape)
# print(b.size)
# print(b.dtype)









#numpy shaping existing array
# import numpy as np

# a = np.arange(1,10)
# print(a, a.ndim)

# shaped = a.reshape((3,3))
# print(shaped)

# reshaped2 = a.reshape((1,9))
# print(reshaped2)










#Concatenation
# import numpy as np

# x = np.array([1,2,3])
# y = np.array([4,5,6])

# concated = np.concatenate([x,y])
# print(concated)
# z = np.array([7,8,9])
# lastConcaneted = np.concatenate([concated, z])
# print(lastConcaneted)

# #2 dimention
# a = np.array([[1,2,3], [4,5,6]])
# concated2 = np.concatenate([a,a])
# print(concated2)
# concated3 = np.concatenate([a,a], axis=1)
# print(concated3)















#array splitting
# import numpy as np

# x = np.array([1,2,3,99,99,3,2,1])
# a,b,c = np.split(x, [3,5])
# print(a,b,c)

#two dimension splitting
# m = np.arange(16).reshape(4,4)
# print(m)
# ust, alt =  np.vsplit(m, [2])
# print(ust, alt)
# left,right = np.hsplit(m, [2])
# print(left,right)











#sorting
# import numpy as np
# v = np.array([2,1,4,3,5])
# print(v)
# print(np.sort(v))
# #if we use v.sort directly it will change v itself
# # two dimension
# matris = np.random.normal(20,5,(3,3))
# print(matris)
# rowSorted = np.sort(matris, axis = 1)
# print(rowSorted)
# columnSorted = np.sort(matris, axis = 0)
# print(columnSorted)










#riching array element
# import numpy as np
# a = np.random.randint(10, size =10)
# print(a, a[0], a[-1])
# a[0] = 1
# print(a)

# b = np.random.randint(10, size = (4,4))
# print(b, b[0][0], b[1][0], b[2][3])

# b[0][0] = 0
# print(b[0][0])

# b[0][1] = 1.5
# print(b)











#riching slicing elements of array
# import numpy as np
# a = np.arange(20,30)
# print(a)
# print(a[0:3])
# print(a[3:-1])
# print(a[3:])
# print(a[1::2])
# print(a[0::3])
#two dimension
# m = np.random.randint(10, size=(5,5))
# print(m)
# print(m[:, 0])
# print(m[:, 1])
# print(m[:, 4])
# print(m[0,:])
# print(m[:,0])
# print(m[0:2,0:3])
# print(m[:,0:3])
# print(m[::,:2])
# print(m[1:3,0:2])













#operations on sub
# import numpy as np

# a = np.random.randint(10, size = (5,5))
# print(a)
# sub_A = a[0:3,0:2]
# print(sub_A)
# sub_A[0,0] = 999
# sub_A[1,1] = 888
# print(sub_A)
# print(a)

#we only want to change sub we add copy 
# a = np.random.randint(10, size = (5,5))
# print(a)
# sub_A = a[0:3,0:2].copy()
# print(sub_A)
# sub_A[0,0] = 999
# sub_A[1,1] = 888
# print(sub_A)
# print(a)















# Fancy index 
# import numpy as np
# v = np.arange(0,30,3)
# print(v)
# print(v[1])
# print(v[3])
# print(v[5])
# print([v[1], v[3], v[5]])
# bring = [1,3,5]
# print(v[bring])

#tow dimension
# m = np.arange(9).reshape((3,3))
# print(m)
# row = np.array([0,1])
# column = np.array([1,2])
# print(row, column)
# print(m[row,column])
# print(m[0, [1,2]])
# print(m[0:, [1,2]])














#conditional element operations
# import numpy as np
# v = np.array([1,2,3,4,5])
# print(v)
# print(v > 3)
# print(v[v<3])
# print(v[v>3])
# print(v[v != 3])
# print(v[v == 3])

# print(v*2)
# print(v/5)
# print(v*5/10-1 + v**2)

















#mathematical operations
# import numpy as np

# v = np.array([1,2,3,4,5,5])
# print(v-1)
# print(v*5)
# print(v%2)
# print(np.mod(v,2))
# print(np.absolute(np.array([-3])))
# print(np.cos(180), np.sin(180))
# print(np.log(np.array([2,3,4])))
# print(np.log2(np.array([2,3,4])))
# print(np.log10(np.array([2,3,4])))

#statistic operations
# import numpy as np
# v = np.array([5,10,20,25,33])
# print(np.mean(v, axis=0))
# print(v.sum())
# print(v.min())
# print(v.max())
# print(np.var(v))
# print(np.std(v))
# print(np.corrcoef(v))













#equation with two unknowns
# import numpy as np
# # 5 * x0 + x1 = 12
# # x0 + 3 * x1 = 10
# a = np.array([[5,1], [1,3]])
# b = np.array([12,10])

# x = np.linalg.solve(a, b)
# print(x)
