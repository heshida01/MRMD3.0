import time

a = time.time()
for x in range(100000000):
    x = x+x
print(time.time()-a)