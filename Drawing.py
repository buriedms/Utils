import matplotlib.pyplot as plt
import numpy as np

t_list=np.linspace(0,1,5)[1:-1]
plt.figure()
for t in t_list:
    x = np.linspace(0, 1, 1002)[1:-1]
    y = -t*np.log(x)-(1-t)*np.log(1-x)
    plt.plot(x,y)
plt.show()

