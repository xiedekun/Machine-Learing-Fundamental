import numpy as np 

class StandardScaler:
    
    def __init__(self):
        self.mean_ = None 
        self.scale_ = None

    def fit(self, X):

        assert X.ndim == 2 ,'只能二维'

        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self 

    def transform(self,X):
        
        assert X.ndim ==2 , ''
        assert self.mean_ is not None and self.scale_ is not None , ''
        assert X.shape[1] == len(self.mean_), '特征数量要和均值方差数量相等'

        resX = np.empty(shape = X.shape, dtype = float)

        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col])/self.scale_[col] 
            
        return resX