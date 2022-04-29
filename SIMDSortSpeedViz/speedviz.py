import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tests = 32
ns = 23

filepaths = glob.glob("../SIMDSort/bin/sortd_*_speed.txt")

datas = []

for filepath in filepaths:
    data = pd.read_csv(filepath, delimiter=',')
    cond = filepath.replace('\\', '/').split('/')[-1][len('sortd_'):-len('_speed.txt')]

    n, stdsort, avxsort = data['n'].to_numpy(), data['std'].to_numpy(), data['avx'].to_numpy()

    n = np.reshape(n, (ns, tests))[:, 0]
    stdsort = np.reshape(stdsort, (ns, tests))
    avxsort = np.reshape(avxsort, (ns, tests))

    # ignore outliner
    stdsort = np.sort(stdsort, axis = 1)[:, tests//16:-tests//16]
    avxsort = np.sort(avxsort, axis = 1)[:, tests//16:-tests//16]
    
    stdsortmean = np.mean(stdsort, axis = 1)
    stdsortsig1 = np.std( stdsort, axis = 1)
    stdsortmax  = np.max( stdsort, axis = 1)
    stdsortmin  = np.min( stdsort, axis = 1)

    avxsortmean = np.mean(avxsort, axis = 1)
    avxsortsig1 = np.std( avxsort, axis = 1)
    avxsortmax  = np.max( avxsort, axis = 1)
    avxsortmin  = np.min( avxsort, axis = 1)

    datas.append((cond, n, avxsortmean))

    fig = plt.figure(figsize=(12, 6))

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('N')
    plt.ylabel('time[usec]')

    plt.xlim([100, 200000000])
    plt.ylim([ 10, 100000000])

    plt.grid()

    plt.plot(n, stdsortmean, label='stdsort intro', color='red')
    plt.scatter(n, stdsortmean, color='red', marker='.')
    plt.fill_between(n, stdsortmean - stdsortsig1, stdsortmean + stdsortsig1, color='red', alpha = 0.4)
    plt.fill_between(n, stdsortmin, stdsortmax, color='red', alpha = 0.2)

    plt.plot(n, avxsortmean, label='avxsort comb -> paracomb -> batch -> scan', color='blue')
    plt.scatter(n, avxsortmean, color='blue', marker='.')
    plt.fill_between(n, avxsortmean - avxsortsig1, avxsortmean + avxsortsig1, color='blue', alpha = 0.4)
    plt.fill_between(n, avxsortmin, avxsortmax, color='blue', alpha = 0.2)

    for i, c in enumerate(n):
        if avxsortmean[i] <= 20:
            continue

        fasterx = stdsortmean[i] / avxsortmean[i]
        plt.text(c, avxsortmean[i] * 0.75, 'x%.2lf' % fasterx, 
                 horizontalalignment='center', verticalalignment='top')
    
    plt.legend(loc='lower right')

    plt.savefig('../figures/sort_{}_speed_d.svg'.format(cond))
    plt.cla()

fig = plt.figure(figsize=(12, 6))

plt.xscale('log')
plt.yscale('log')

plt.xlabel('N')
plt.ylabel('time[usec]')

plt.xlim([100, 200000000])
plt.ylim([ 10, 100000000])

plt.grid()

for (cond, n, speed) in datas:
    plt.plot(n, speed, label=cond)

plt.legend(loc='lower right')

plt.savefig('../figures/sort_avxall_speed_d.svg')
plt.cla()