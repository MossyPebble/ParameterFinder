import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import joblib
from decimal import Decimal

mlr = joblib.load('mlrmodel.pkl')



def cartesian(arrays):

    """[[numpy 배열], [numpy 배열]...] 의 형태를 받아 배열 끼리의 데카르트 곱을 numpy 배열로 반환"""

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

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
    
def equationans(Tox, muns, Vt, m):
    temp = cartesian((np.arange(0, 5), np.arange(0, 5)))
    return [float(equation(Tox, muns, Vt, m, *x)) for x in temp]
    
midTox = Decimal('1e-8')
midmuns = Decimal('0.02')
midm = Decimal('1.2')
midVt = Decimal('0.7')

test = mlr.predict([equationans(midTox, midmuns, midm, midVt)])
print(test)