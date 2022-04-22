import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tests = 32

data = pd.read_csv("../SIMDSort/bin/sort_random_speed.txt", delimiter=',')

n, stdsort, avxsort = data['n'].to_numpy(), data['std'].to_numpy(), data['avx'].to_numpy()

n = np.reshape(n, (-1, tests))[:, 0]
stdsort = np.reshape(stdsort, (-1, tests))
avxsort = np.reshape(avxsort, (-1, tests))

stdsortmean = np.mean(stdsort, axis = 1)
stdsortsig1 = np.std( stdsort, axis = 1)
stdsortmax  = np.max( stdsort, axis = 1)
stdsortmin  = np.min( stdsort, axis = 1)

avxsortmean = np.mean(avxsort, axis = 1)
avxsortsig1 = np.std( avxsort, axis = 1)
avxsortmax  = np.max( avxsort, axis = 1)
avxsortmin  = np.min( avxsort, axis = 1)

fastor = stdsortmean / avxsortmax

fastor_sort = n[np.argsort(fastor)]

fig = plt.figure(figsize=(12, 6))

plt.xscale('log')
plt.yscale('log')

plt.xlabel('N')
plt.ylabel('time[usec]')

plt.xlim([100, 200000000])
plt.ylim([ 10, 100000000])

plt.grid()

plt.plot(n, stdsortmean, label='stdsort quick', color='red')
plt.fill_between(n, stdsortmean - stdsortsig1, stdsortmean + stdsortsig1, color='red', alpha = 0.4)
plt.fill_between(n, stdsortmin, stdsortmax, color='red', alpha = 0.2)

plt.plot(n, avxsortmean, label='avxsort comb -> backtrack -> scan', color='blue')
plt.fill_between(n, avxsortmean - avxsortsig1, avxsortmean + avxsortsig1, color='blue', alpha = 0.4)
plt.fill_between(n, avxsortmin, avxsortmax, color='blue', alpha = 0.2)

plt.legend(loc='lower right')

plt.savefig('../figures/random_speed.svg')