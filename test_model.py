import unittest
import numpy as np
import dill
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter
from logisticReg import LogisticRegression


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
