import numpy as np
a = np.zeros(2)
a[0] = 1
a[1] = 2
print (int(np.where(a == np.max(a))[0]))