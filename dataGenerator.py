import numpy as np
import time
import tensorflow as tf
import pandas as pd
import random

def equation(Tox, muns, Vgs, Vt, m, Vds, W=1e-6, L=1e-6, Eox=3.9 * 8.85e-12):
    Coxe = Eox/Tox
    if Vds < (Vgs - Vt)/m:
        return W/L * Coxe * muns * (Vgs-Vt-m/2*Vds)*Vds
    elif Vds >= (Vgs - Vt)/m:
        return W/(2*m*L) * Coxe * muns * (Vgs - Vt)*(Vgs - Vt)
    elif Vgs < Vt:
        return 0
    else:
        raise Exception('방정식 예외 발생')
    
midTox = 1000 #1e-8
midmuns = 2000 #0.02
midm = 12000 #1.2
midVt = 7000 #0.7

data = None
ans = None

def cartesian(self, arrays):

    """[[numpy 배열], [numpy 배열]...] 의 형태를 받아 배열 끼리의 데카르트 곱을 numpy 배열로 반환"""

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

def splitGenerator(*args):

    """args는 튜플 (시작, 끝, 간격)으로 구성, 받은 모든 args를 해당하는 간격으로 잘라 self.data에 저장"""
    
    title = np.array([np.array(['Tox', 'muns', 'Vgs', 'Vt', 'm', 'Vds', 'answer'])])
    case = [np.arange(*[int(x) for x in i]) for i in args]
    caselist = cartesian(case)
    ans = np.array([np.array([equation(*x)]) for x in caselist])
    data = np.random.permutation(np.concatenate((caselist, ans), axis=1))
    data = np.concatenate((title, data), axis=0)
    return data

def randomGenerator(*args, cnt):

    """args는 튜플 (시작, 끝)으로 구성, 받은 모든 args안의 영역의 랜덤한 데이터를 cnt 횟수 만큼 채워 넣음"""

    title = np.array([np.array(['Tox', 'muns', 'Vgs', 'Vt', 'm', 'Vds', 'answer'])])
    case = [np.arange(*[int(x) for x in i], dtype=int) for i in args]
    tempcaselist = []
    for i in range(0, cnt):
        tempcaselist.append(np.array([x[random.randrange(0, len(case))] for x in case]))
    caselist = np.array(tempcaselist)
    ans = np.array([np.array([equation(*x)]) for x in caselist])
    data = np.random.permutation(np.concatenate((caselist, ans), axis=1))
    data = np.concatenate((title, data), axis=0)
    return data

def dataSave():
    pd.DataFrame(data).to_csv("data.csv", index=False, header=False)

def dataPrint():
    print(data)





start = time.time()
data = randomGenerator((midTox*0.7, midTox*1.3, midTox*0.6/100), (midmuns*0.7, midmuns*1.3, midmuns*0.6/100), (0, 50, 1), (midVt*0.7, midVt*1.3, midVt*0.6/100), (midm*0.7, midm*1.3, midm*0.6/100), (0, 50, 1), cnt=1000000)
dataSave()
dataPrint()
print(float(time.time() - start))