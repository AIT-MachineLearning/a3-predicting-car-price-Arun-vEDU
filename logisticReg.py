# -*- coding: utf-8 -*-
"""Deployment - Copy.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1erlvNieUJ1D4qB3ckRT6FE_chL9FnlGq
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#load data
df = pd.read_csv('C:\\Git\\a3-predicting-car-price-Arun-vEDU\\Cars_a3.csv')

#Step 1: Prepare data
# y is simply the selling price colomn
y = df["selling_price"]

# Covert into log scale
y_log = np.log(df["selling_price"])


# Using pd.cut to bin data into 4 classes
binned_data = pd.cut(y_log , bins=4) #now our y is four classes thus require multinomial

# Prepare data
# y is simply the selling price colomn
y = df["selling_price"]

# Covert into log scale
y_log = np.log(df["selling_price"])


binned_data = pd.cut(y_log, bins=4, labels=False)  # Creates 0, 1, 2, 3
# plot the values
# Value counts for each bin
bin_counts = pd.value_counts(binned_data)

# Add binned_data as a new column in the DataFrame
df['binned_price'] = binned_data

# Step 3: Prepare features and labels
X = df.drop(columns=['selling_price', 'binned_price'])
y = df['binned_price']  # Original binned labels (0, 1, 2, 3)

#print(binned_data)

# my selected features are 'Max Power' and 'Year' from a1 and a2 assignments.

# Split the 'max_power' column by space
df[['max_power', 'max_power_unit']] = df['max_power'].str.split(' ', expand=True)

# Remove the 'max_power_unit' column
df = df.drop('max_power_unit', axis=1)
df.columns

# Convert 'max_power' to a numeric type
df['max_power'] = pd.to_numeric(df['max_power'])

# To check the data types of the columns
#print(df)
#print(df.dtypes)

# X is my selected strong features
# I have selected 2 features "year" and "max_power"
# less features are better.
X = df[        ['year', 'max_power']        ]

#Split the data into train and test
# test set is 20% of our dataset.
# This is a medium size data set, that is why it is selected 80%  traning set and 20% test set

from sklearn.model_selection import train_test_split
#  X is your feature set and binned_data is your target (with 4 classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#check for null values in traning set,The .isna() method is used to detect missing values (not avalible) and count it
X_train[['year', 'max_power']].isna().sum()
#check for null values in test set, and count it
X_test[['year', 'max_power']].isna().sum()

# max power has 167 missing values. So we can fill them with ratio, the distribution of the 'max_power' is skewed.
# X_train is  training dataset and 'max_power' contains missing values

# Step 1: Calculate the ratio of each unique value in 'max_power'
ratio = X_train['max_power'].value_counts(normalize=True)

# Step 2: Define a function to sample values based on these ratios
def fill_missing_values(column, value_distribution):
    missing_indices = column[column.isnull()].index  # Find indices where values are missing
    filled_values = np.random.choice(value_distribution.index,
                                     size=len(missing_indices),
                                     p=value_distribution.values)  # Sample from distribution

    # Fill the missing values with sampled values
    column.loc[missing_indices] = filled_values
    return column

# Step 3: Fill missing values in 'max_power' based on the ratio
X_train['max_power'] = fill_missing_values(X_train['max_power'], ratio)

# Now, X_train['max_power'] has the missing values filled based on the value distribution
print(X_train['max_power'].isnull().sum())  # This should return 0 (no missing values)

# Step 4: Fill missing values in 'max_power' in X_test using the same ratio as X_train
X_test['max_power'] = fill_missing_values(X_test['max_power'], ratio)

# Now, X_test['max_power'] should have the missing values filled based on the distribution from X_train
print(X_test['max_power'].isnull().sum())  # This should also return 0 (no missing values)

# I'm running the custom class for evaluate bast parameters with out mlflow.
from collections import Counter
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# Step 1: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scaled X_train
X_test_scaled = scaler.transform(X_test)        # Scaled X_test

y_labels = y_train  # Assuming y_train contains the labels (0, 1, 2, 3)

# Step 2: Apply SMOTE to the scaled X_train
smote = SMOTE(random_state=42)
X_train_smote, y_labels_smote = smote.fit_resample(X_train_scaled, y_labels)  # Use the scaled X_train

# Check the class distribution after SMOTE
print("After SMOTE:", Counter(y_labels_smote))

# One-hot encode the labels
y_labels_smote_one_hot = pd.get_dummies(y_labels_smote, prefix='class')

# Step 3: Add intercept term AFTER SMOTE and scaling
intercept_train = np.ones((X_train_smote.shape[0], 1))  # Shape (m, 1)
X_train_with_intercept = np.concatenate((intercept_train, X_train_smote), axis=1)  # Shape (m, n + 1)

intercept_test = np.ones((X_test_scaled.shape[0], 1))  # Shape (m, 1)
X_test_with_intercept = np.concatenate((intercept_test, X_test_scaled), axis=1)  # Shape (m, n + 1)

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