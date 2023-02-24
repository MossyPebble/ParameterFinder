import numpy as np
import time
import tensorflow as tf
import pandas as pd
import random
from decimal import Decimal

def cartesian(arrays):

    """[[numpy 배열], [numpy 배열]...] 의 형태를 받아 배열 끼리의 데카르트 곱을 numpy 배열로 반환"""

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

def splitGenerator(*args):

    """args는 튜플 (시작, 끝, 간격)으로 구성, 받은 모든 args를 해당하는 간격으로 잘라 self.data에 저장"""
    
    title = np.array([np.array(['Tox', 'muns', 'Vt', 'm', 'Vds', 'Vgs', 'answer'])])
    case = [np.arange(*[x for x in i]) for i in args]
    caselist = cartesian(case)
    ans = np.array([np.array([equation(*x)]) for x in caselist])
    data = np.random.permutation(np.concatenate((caselist, ans), axis=1))
    data = np.concatenate((title, data), axis=0)
    return data

def randomGenerator(*args, cnt):

    """args는 튜플 (시작, 끝)으로 구성, 받은 모든 args안의 영역의 랜덤한 데이터를 cnt 횟수 만큼 채워 넣음"""

    title = np.array([np.array(['Tox', 'muns', 'Vt', 'm', 'Vds', 'Vgs', 'answer'])])
    case = [np.arange(*[x for x in i]) for i in args]
    tempcaselist = []
    for i in range(0, cnt):
        print(np.array([x[random.randrange(0, len(case))] for x in case]))
        tempcaselist.append(np.array([x[random.randrange(0, len(case))] for x in case]))
    caselist = np.array(tempcaselist)
    ans = np.array([np.array([equation(*x)]) for x in caselist])
    data = np.random.permutation(np.concatenate((caselist, ans), axis=1))
    data = np.concatenate((title, data), axis=0)
    return data

def dataSave(data):
    pd.DataFrame(data).to_csv("data.csv", index=False, header=False)

def equation(Tox, muns, Vt, m, Vds, Vgs, W=Decimal('1e-6'), L=Decimal('1e-6'), Eox=Decimal('3.9')*Decimal('8.85e-12')):
    Coxe = Eox/Tox
    if Vds < (Vgs - Vt)/m:
        return W/L * Coxe * muns * (Vgs-Vt-m/2*Vds)*Vds
    elif Vds >= (Vgs - Vt)/m:
        return W/(2*m*L) * Coxe * muns * (Vgs - Vt)*(Vgs - Vt)
    elif Vgs < Vt:
        return 0
    else:
        raise Exception('방정식 예외 발생')
    
midTox = Decimal('1e-8')
midmuns = Decimal('0.02')
midm = Decimal('1.2')
midVt = Decimal('0.7')



start = time.time()
data = randomGenerator(*[(x*Decimal('0.7'), x*Decimal('1.3'), x*Decimal('0.6')/Decimal('100')) for x in [midTox, midmuns, midm, midVt]], (0, 50, 1), (0, 50, 1), cnt=30)
dataSave(data)
print(float(time.time() - start))