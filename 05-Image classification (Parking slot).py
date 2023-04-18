# Data -> https://drive.google.com/file/d/11DyZ165lZGzULEZSQNofyy9A8xaYgFJ2/view

import os
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Problem to solve: Find a classifier to split between free / occupied parking slot image

# Get & label data
input_dir = 'C:/Users/coronis/Pictures/clf-data'
categories = ['empty', 'not_empty']
data = []
labels = []
for idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())  # BGR array to unidimensional array
        labels.append(idx)

data = np.asarray(data)
labels = np.asarray(labels)

# Train / test split
# 80% Train set 20% Test set. Shuffle data before split. Stratify will keep the same proportion in our split sets.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train classifier
# Support vector classifier
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C':[1, 10, 100, 1000]}]  # 12 possible classifiers
# Optimal parameters search
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(X_train, y_train)

# Obtained model evaluation
best_classifier = grid_search.best_estimator_
y_predictions = best_classifier.predict(X_test)
score = accuracy_score(y_predictions, y_test)
print('{}% Accuracy score'.format(str(score * 100)))

# Save obtained model
# pickle.dump(best_classifier, open('./model.p', 'wb'))