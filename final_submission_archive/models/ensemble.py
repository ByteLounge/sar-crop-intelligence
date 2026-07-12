import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor

class OptimizedCropEnsemble:
    """
    Kaggle-optimized ensemble incorporating specific tuned models and weights per crop.
    """
    def __init__(self, target: str, random_state: int = 42):
        self.target = target
        if target == 'Rice_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 1.0, 0.0
        elif target == 'Cotton_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.05, random_seed=random_state, verbose=0)
            self.w1, self.w2 = 0.8, 0.2
        elif target == 'Maize_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 0.0, 1.0
        elif target == 'Bajra_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 0.0, 1.0
        elif target == 'Groundnut_frac':
            self.model1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model2 = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            self.w1, self.w2 = 1.0, 0.0
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.model1.predict(X)
        p2 = self.model2.predict(X)
        return self.w1 * p1 + self.w2 * p2
