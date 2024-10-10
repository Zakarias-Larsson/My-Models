import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, n_iters=1000, regularization=None, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.regularization = regularization
        self.lambda_param = lambda_param

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights_) + self.bias_
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            if self.regularization == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights_
            elif self.regularization == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights_)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        linear_model = np.dot(X, self.weights_) + self.bias_
        y_predicted = self._sigmoid(linear_model)
        return (y_predicted > 0.5).astype(int)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # Load the German Credit dataset from OpenML
    X, y = fetch_openml(name='credit-g', version=1, as_frame=True, return_X_y=True)

    # Preprocessing: Handle missing values and encode categorical features
    # Impute missing values (if any)
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)

    # Convert categorical data to numerical values
    X_encoded = pd.get_dummies(pd.DataFrame(X_imputed, columns=X.columns), drop_first=True)

    # Encode target variable (y) as it is categorical ('good'/'bad')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the logistic regression model
    model = LogisticRegression(learning_rate=0.001, n_iters=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Check the coefficients (weights) of the model
    print("Feature Coefficients:")
    print(model.weights_)

