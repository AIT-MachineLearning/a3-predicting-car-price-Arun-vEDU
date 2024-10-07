import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter
import time
# Logistic Regression class
class LogisticRegression:
    def __init__(self, k, n, method, alpha=0.1, max_iter=5000, use_penalty=False, penalty='ridge', lambda_=0.01):
        self.k = k  # Number of classes
        self.n = n  # Number of features
        self.alpha = alpha  # Learning rate
        self.max_iter = max_iter  # Maximum iterations
        self.method = method  # Optimization method: 'batch', 'minibatch', or 'sto'
        self.use_penalty = use_penalty  # Whether to use penalty (regularization)
        self.penalty = penalty  # Type of penalty ('ridge' for L2)
        self.lambda_ = lambda_  # Regularization strength
        self.W = np.random.rand(n + 1, k)  # Initialize weights

    def fit(self, X, Y):
        self.W = np.random.rand(self.n + 1, self.k)  # Initialize weights
        self.losses = []  # To store loss values
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W -= self.alpha * grad  # Update weights
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss}")
            print(f"Time taken: {time.time() - start_time}")
        
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                indices = np.random.choice(X.shape[0], size=batch_size, replace=False)  # Randomly select indices for the batch
                batch_X = X[indices]
                batch_Y = Y[indices]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W -= self.alpha * grad  # Update weights
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss}")
            print(f"Time taken: {time.time() - start_time}")
        
        elif self.method == "sto":
            start_time = time.time()
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])  # Randomly select an index
                X_train = X[idx, :].reshape(1, -1)  # Reshape for a single sample
                Y_train = Y[idx].reshape(1, -1)  # Reshape for a single sample
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W -= self.alpha * grad  # Update weights
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss}")
            print(f"Time taken: {time.time() - start_time}")
        
        else:
            raise ValueError('Method must be one of the following: "batch", "minibatch", or "sto".')
        
    def gradient(self, X, Y):
        m = X.shape[0]  # Number of training examples
        h = self.h_theta(X, self.W)  # Hypothesis
        loss = -np.sum(Y * np.log(h)) / m  # Cross-entropy loss
        
        # Apply penalty if use_penalty is True
        if self.use_penalty and self.penalty == 'ridge':
            loss += (self.lambda_ / (2 * m)) * np.sum(np.square(self.W))  # Ridge penalty (L2)
        
        error = h - Y  # Error term
        grad = self.softmax_grad(X, error)
        
        # Apply gradient for penalty if use_penalty is True
        if self.use_penalty and self.penalty == 'ridge':
            grad += (self.lambda_ / m) * self.W  # Add Ridge gradient (L2)
        
        return loss, grad
        
    def softmax(self, theta_t_x):
        # Ensure input is a NumPy array
        theta_t_x = np.array(theta_t_x)
    
        # Perform softmax calculation
        exp_values = np.exp(theta_t_x - np.max(theta_t_x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error / X.shape[0]

    def h_theta(self, X, W):
        return self.softmax(X @ W)
    
    def predict(self, X_test):
            h = self.h_theta(X_test, self.W)  # Get the probabilities
            return np.argmax(h, axis=1)  # Return the predicted class labels directly

    def f1_score(self, y_true, y_pred, class_label):
        prec = self.precision(y_true, y_pred, class_label)
        rec = self.recall(y_true, y_pred, class_label)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def precision(self, y_true, y_pred, class_label):
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        FP = np.sum((y_true != class_label) & (y_pred == class_label))
        return TP / (TP + FP) if (TP + FP) > 0 else 0

    def recall(self, y_true, y_pred, class_label):
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        FN = np.sum((y_true == class_label) & (y_pred != class_label))
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def macro_f1(self, y_true, y_pred):
        classes = np.unique(y_true)
        f1_scores = [self.f1_score(y_true, y_pred, class_label) for class_label in classes]
        return np.mean(f1_scores)


class TestLogisticRegressionModel(unittest.TestCase):

    def setUp(self):
        # Create an instance of the model before each test
        self.model = LogisticRegression(k=4, n=2, method='batch')
        
        # Prepare mock training data
        self.X_train = np.random.rand(100, 2)  # 100 samples, 2 features
        self.y_train = np.random.randint(0, 4, 100)  # Random labels for 4 classes

        # Step 1: Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)  # Scaled X_train

        # Step 2: Apply SMOTE to the scaled X_train
        self.smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_labels_smote = self.smote.fit_resample(self.X_train_scaled, self.y_train)

        # Check the class distribution after SMOTE
        print("After SMOTE:", Counter(self.y_labels_smote))

        # One-hot encode the labels
        self.y_labels_smote_one_hot = pd.get_dummies(self.y_labels_smote, prefix='class').values

        # Step 3: Add intercept term AFTER SMOTE and scaling
        self.intercept_train = np.ones((self.X_train_smote.shape[0], 1))  # Shape (m, 1)
        self.X_train_with_intercept = np.concatenate((self.intercept_train, self.X_train_smote), axis=1)  # Shape (m, n + 1)

    def test_input_shape(self):
        # Test if the model takes the expected input shape
        try:
            self.model.fit(self.X_train_with_intercept, self.y_labels_smote_one_hot)
        except ValueError as e:
            self.fail(f"Model raised ValueError unexpectedly: {e}")

    def test_output_shape(self):
        # Test if the output of the model has the expected shape after fitting
        self.model.fit(self.X_train_with_intercept, self.y_labels_smote_one_hot)
        predictions = self.model.predict(self.X_train_with_intercept)

        # Expected output shape should be the same number of samples as input
        self.assertEqual(predictions.shape, (self.X_train_with_intercept.shape[0],))  # Check if predictions have shape (m,)

if __name__ == '__main__':
    unittest.main()
