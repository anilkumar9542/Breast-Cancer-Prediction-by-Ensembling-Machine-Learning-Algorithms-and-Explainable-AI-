#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

# for displaying all features from the dataset:
pd.pandas.set_option('display.max_columns', None)

# Reading Dataset:
dataset = pd.read_csv("Breast_data.csv")

# Dropping 'id' and 'Unnamed: 32' features:
dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)

# Encoding on target feature:
dataset['diagnosis'] = np.where(dataset['diagnosis'] == 'M', 1, 0)

# Splitting Independent and Dependent Feature:
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# Feature Importance:
model = ExtraTreesClassifier()
model = model.fit(X, y)

# Displaying feature importance:
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importance:")
print(feature_importance)

# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

# Listing out highly correlated features:
correlated_features = list(correlation(dataset, 0.7))

# Dropping highly correlated features:
X = X.drop(correlated_features, axis=1)
print(X.columns.tolist())

# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# RandomForestClassifier:
random_forest = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5, min_samples_leaf=5, max_features=2)
random_forest = random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)




# Gradient Boosting Classifier:
gradient_boosting = GradientBoostingClassifier(random_state=42,n_estimators=50, max_depth=5, min_samples_leaf=5, max_features=2)
gradient_boosting = gradient_boosting.fit(X_train, y_train)
y_pred_gb = gradient_boosting.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)


# Create a voting classifier with soft voting
# Define base models
random_forest = RandomForestClassifier(random_state=42, n_estimators=100)
gradient_boosting = GradientBoostingClassifier(random_state=42, n_estimators=100)

# Hyperparameter tuning for Random Forest
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3)
grid_rf.fit(X_train, y_train)

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3)
grid_gb.fit(X_train, y_train)

# Update the base models with the best parameters
random_forest = grid_rf.best_estimator_
gradient_boosting = grid_gb.best_estimator_

# Create a voting classifier with soft voting
ensemble_classifier = VotingClassifier(estimators=[
    ('random_forest', random_forest),
    ('gradient_boosting', gradient_boosting),
], voting='soft')

# Train the ensemble classifier
ensemble_classifier.fit(X_train, y_train)

# Make predictions
ensemble_predictions = ensemble_classifier.predict(X_test)

# Calculate Accuracy
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)




# Creating a pickle file for the best model
filename = 'ensemble_classifier(75-25).pkl'
pickle.dump(ensemble_classifier, open(filename, 'wb'))
