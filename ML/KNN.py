import numpy as np 
from math import sqrt 
from collections import Counter 

def kNN_classify(k,X_train,y_train,x):
    assert 1<=k<=X_train.shape[0] , 'k must be valid'
    assert X_train.shape[0] ==y_train.shape[0],\
        'size'
    assert X_train.shape[1] ==x.shape[0],\
        'size'
    distances = [sqrt(np.sum((x_train2 - x)**2)) for x_train2 in X_train]
    nearest = np.argsort(distances)
    topk_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topk_y)

    return votes.most_common(1)[0][0]