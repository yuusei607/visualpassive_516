import random
import matplotlib.pyplot as plt
N_list = [10,50,100,500]
count_list = []
for N in N_list:
    now = set()
    com = set(range(1,N+1))
    count = 0
    
    while now != com:
        now.add(random.randint(1,N))
        count += 1
    count_list.append(count)

plt.plot(N_list,count_list)
plt.show()