import unittest
import numpy as np
from test_copy import LogisticRegression  # Replace with the actual module name

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.k = 4  # Number of classes
        self.n = 2  # Number of features
        self.model = LogisticRegression(k=self.k, n=self.n, method='batch')

    def test_input_shape(self):
        X = np.random.rand(100, self.n)  # Create dummy data
        y = np.random.randint(0, self.k, 100)  # Create dummy labels
        self.model.fit(X, y)  # Fit the model
        
        # Check if the model has been fitted by verifying the weights shape
        self.assertEqual(self.model.W.shape, (self.n + 1, self.k), "Model weights do not have the expected shape after fitting.")

    def test_output_shape(self):
        X_test = np.random.rand(20, self.n)  # Create dummy test data
        self.model.fit(np.random.rand(100, self.n), np.random.randint(0, self.k, 100))  # Fit the model first
        
        y_pred = self.model.predict(X_test)  # Make predictions
        self.assertEqual(y_pred.shape, (20,), "Output predictions do not have the expected shape.")

if __name__ == '__main__':
    unittest.main()
