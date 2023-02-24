import numpy as np
import time
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

def randomGenerator(*args, cnt):

    """args는 튜플 (시작, 끝)으로 구성, 받은 모든 args안의 영역의 랜덤한 데이터를 cnt 횟수 만큼 채워 넣음"""

    datatitle = np.array([np.array(['Tox', 'muns', 'Vt', 'm'])])
    temp = cartesian((np.arange(0, 5), np.arange(0, 5)))
    anstitle = np.array([np.array([str(x) for x in temp])])
    case = [np.arange(*[x for x in i]) for i in args]
    tempcaselist = []
    for i in range(0, cnt):
        tempcaselist.append(np.array([x[random.randrange(0, len(x))] for x in case]))
    caselist = np.array(tempcaselist)
    
    ans = []
    for i in caselist:
        ans.append(np.array([equation(*i, *x) for x in temp]))
    ans = np.array(ans)
    print(ans)
    data = np.concatenate((datatitle, caselist), axis=0)
    ans = np.concatenate((anstitle, ans), axis=0)
    return data, ans

def dataSave(data, ans):
    pd.DataFrame(data).to_csv("data.csv", index=False, header=False)
    pd.DataFrame(ans).to_csv("ans.csv", header=False, index=False)

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
data, ans = randomGenerator(*[(x*Decimal('0.7'), x*Decimal('1.3'), x*Decimal('0.6')/Decimal('1000')) for x in [midTox, midmuns, midm, midVt]], cnt=100000)
dataSave(data, ans)
print(float(time.time() - start))