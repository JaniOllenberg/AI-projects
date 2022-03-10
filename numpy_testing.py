import numpy as np
print(np.array([2,4]))
data = np.array([1, 2])
data2 = np.array([(3, 4), (4, 5)])
print(data)
data3 = np.vstack((data,data2))
print(data3)

print(np.empty((1,3)))
print(len(data3))