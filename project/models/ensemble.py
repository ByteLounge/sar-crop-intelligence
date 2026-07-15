import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

class OptimizedCropEnsemble:
    """
    Kaggle-optimized ensemble incorporating specific tuned models and weights per crop.
    """
    def __init__(self, target: str, random_state: int = 42):
        self.target = target
        if target == 'Rice_frac':
            self.model1 = Ridge(alpha=0.1, random_state=random_state)
            self.model2 = None
            self.w1, self.w2 = 1.0, 0.0
        elif target == 'Cotton_frac':
            self.model1 = BayesianRidge()
            self.model2 = None
            self.w1, self.w2 = 1.0, 0.0
        elif target == 'Maize_frac':
            self.model1 = Ridge(alpha=0.01, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1)
            self.w1, self.w2 = 0.95, 0.05
        elif target == 'Bajra_frac':
            self.model1 = ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1)
            self.w1, self.w2 = 0.95, 0.05
        elif target == 'Groundnut_frac':
            self.model1 = KNeighborsRegressor(n_neighbors=3)
            self.model2 = None
            self.w1, self.w2 = 1.0, 0.0
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model1.fit(X, y)
        if self.model2 is not None:
            self.model2.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.model1.predict(X)
        if self.model2 is not None:
            p2 = self.model2.predict(X)
            return self.w1 * p1 + self.w2 * p2
        return p1
