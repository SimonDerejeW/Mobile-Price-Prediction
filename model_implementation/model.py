import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class GradientDescentMultipleRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None
        self.intercept = None
        self.cost_history = []
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        self.cost_history = []
        
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
            
            cost = (1/(2*n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
        return self
    
    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        return X @ self.coefficients + self.intercept
    
    def find_best_learning_rate(self, X_train, y_train, X_val, y_val, learning_rates=None):
        """
        Find best learning rate using validation data to prevent overfitting
        """
        if learning_rates is None:
            # Test more learning rates in a reasonable range
            # Creates 20 points between 0.0001 and 1
            learning_rates = [1e-7, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]
            
        results = {}
        best_lr = None
        best_val_r2 = float('-inf')

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        print("Learnings rates to test:", learning_rates)
        
        for lr in learning_rates:
            print('learning rate', lr)
            # Store original learning rate
            original_lr = self.learning_rate
            self.learning_rate = lr
            
            # Fit model
            self.fit(X_train, y_train)
            
            # Get predictions on validation set
            val_pred = self.predict(X_val)
            

            # Calculate metrics
            val_r2 = r2_score(y_val, val_pred)

            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            results[lr] = {
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'cost_history': self.cost_history.copy()
            }
            
            # Update best learning rate based on validation R²
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_lr = lr
            
            # Restore original learning rate
            self.learning_rate = original_lr
            
        return best_lr, results