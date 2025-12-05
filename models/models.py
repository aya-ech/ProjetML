import numpy as np
from sklearn.exceptions import NotFittedError

class LinearSVCScratch:
    def __init__(self, lr=0.001, n_iters=1000, C=None):
        """
        lr: learning rate
        n_iters: number of iterations
        C: regularization parameter. 
           C=None -> hard-margin, C>0 -> soft-margin
        """
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.w = None
        self.b = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,) with values -1 or 1
        """
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # No misclassification
                    dw = self.w
                    db = 0
                else:
                    # Misclassified
                    if self.C is None:  # Hard-margin
                        dw = self.w - y_[idx] * x_i
                        db = -y_[idx]
                    else:  # Soft-margin
                        dw = self.w - self.C * y_[idx] * x_i
                        db = -self.C * y_[idx]

                # Gradient descent update
                self.w -= self.lr * dw
                self.b -= self.lr * db
        self.fitted_ = True
        return self

    def predict(self, X):
        if not self.fitted_:
            raise NotFittedError("This LinearSVM instance is not fitted yet.")
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.fitted_ = False

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db
        self.fitted_ = True
        return self

    def predict_prob(self, X):
        if not self.fitted_:
            raise NotFittedError("This LogisticRegression instance is not fitted yet.")
        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        probs = self.predict_prob(X)
        return np.where(probs >= 0.5, 1, 0)


class KNN_Scratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.fitted_ = False

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.fitted_ = True
        return self

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        if not self.fitted_:
            raise NotFittedError("This KNN instance is not fitted yet.")
            
        y_pred = []

        for x in X:
            # Compute distances from x to all training points
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            # Get indices of k nearest neighbors
            k_idx = np.argsort(distances)[:self.k]
            # Voting
            k_neighbor_labels = [self.y_train[i] for i in k_idx]
            label = max(set(k_neighbor_labels), key=k_neighbor_labels.count)
            y_pred.append(label)

        return np.array(y_pred)
