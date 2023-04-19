# Dataset -> https://www.kaggle.com/datasets/camnugent/california-housing-prices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Regressor not classifier

# Load dataset
dataset = pd.read_csv("housing.csv")
print("Dataset:\n", dataset)

# Pre-processing
# We need to preprocess text values to numerical inputs to feed the model
dataset.dropna(inplace=True)  # Remove null values. Inplace -> Modify current dataset, doesn't create a new one
dataset['bedroom_ratio'] = dataset['total_bedrooms'] / dataset['total_rooms']  # Bedrooms per house
dataset['households_rooms'] = dataset['total_rooms'] / dataset['households']  # Rooms per household

print("\nWe have this ocean_proximity possible values:\n", dataset['ocean_proximity'].value_counts())
print("\nIf we have more than 2 categories, instead of assign numbers to all of them. Create new categories to cover "
      "all possible values and assign 0 or 1")
    # label_quality = LabelEncoder()  # text values to numbers
    # dataset['ocean_pr oximity'] = label_quality.fit_transform(dataset['ocean_proximity'])
    # print("Now we have this possible values:", dataset['ocean_proximity'].unique())
dataset = dataset.join(pd.get_dummies(dataset['ocean_proximity'])).drop(['ocean_proximity'], axis=1)

dataset['total_rooms'] = np.log(dataset['total_rooms'] + 1)  # To avoid 0 values and get log norm distribution
dataset['total_bedrooms'] = np.log(dataset['total_bedrooms'] + 1)  # To avoid 0 values and get log norm distribution
dataset['population'] = np.log(dataset['population'] + 1)  # To avoid 0 values and get log norm distribution
dataset['households'] = np.log(dataset['households'] + 1)  # To avoid 0 values and get log norm distribution
dataset['median_income'] = np.log(dataset['median_income'] + 1)  # To avoid 0 values and get log norm distribution
print("Our refactored dataset:\n", dataset)


# Analysis
# Histogram of the distribution of our numerical features
dataset.hist(figsize=(20, 10))
plt.show()
# Visualize the correlation between features
plt.figure(figsize=(20, 10))
sns.heatmap(dataset.corr(), annot=True, cmap="YlGnBu")
plt.show()

# Total rooms have more impact than total bedrooms
# Bedroom ratio has a negative impact for median house value
# Households have 0 correlation approx. with the target value
# Households rooms have more impact with the target value
# We will use all the data to train our model, but we can drop data that has approx. 0 correlation with target value

# Visualize house position and price correlation
# Scatterplot -> (Dispersion diagram or point graphic) it's a graphical representation to show relationship between
# two variables represented in X and Y axis. We can add a third dimension (bubble graphic) where the first two values
# are represented in the same way and the third it's represented with the bubble size or color.
plt.figure(figsize=(20, 10))
sns.scatterplot(x="latitude", y="longitude", data=dataset, hue="median_house_value", palette="coolwarm")
plt.show()


# Training models
# Train & test split
X = dataset.drop(['median_house_value'], axis=1)  # Axis -> Drop row or column
y = dataset['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data = X_train.join(y_train)

# Scale the data, all features must be taken in account (we don't need to scale the outputs)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Linear regression model
# Linear regression algorithm searches the best line that fits the input data. (Not usable in classification)
reg = LinearRegression()
reg.fit(X_train, y_train)
print("\nLinear regression score:", reg.score(X_test, y_test))

# Random forest model
# Random forest algorith creates decision trees using different random subsets of the data and features.
# Each decision tree provides its opinion on how to classify the data.
# Predictions are made by taking the most popular result.
forest = RandomForestRegressor(n_estimators=200)
forest.fit(X_train, y_train)
print("\nRandom forest regressor score:", forest.score(X_test, y_test))

# Optimize random forest
forest2 = RandomForestRegressor(n_estimators=200)
param_grid = {'n_estimators': [100, 200, 300], 'min_samples_split': [2, 4], 'max_depth': [None, 4, 8]}
grid_search = GridSearchCV(forest2, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(X_train, y_train)
print("Our new Random forest regressor best estimator:", grid_search.best_estimator_)
print("Our new Random forest regressor best param:", grid_search.best_params_)
print("Our new Random forest regressor best score:", grid_search.best_score_)  # Best score
print("Our new Random forest regressor mean score:", grid_search.score(X_test, y_test)) # Mean score