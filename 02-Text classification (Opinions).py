import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import SVC

# Problem to solve: Find a classifier to split opinions between camera and car text opinions

# Analyzing data
# Dataset -> https://www.kaggle.com/datasets/jyotiprasadpal/eopinionscom-product-reviews
print("\nReading dataset Eopinions.csv")
df = pd.read_csv('Eopinions.csv')
print(df)
pd.set_option("display.max_colwidth", None)  # Display without width limit to print
print(df.loc[0])  # Display only first row

pd.set_option("display.max_colwidth", 50)   # Back to default display
print(df.loc[0])

print("\nPrint number of words in each text row:")
print(df['text'].str.count(' '))  # Count whitespaces
print(df['text'].str.split().str.len())  # Count whitespaces
print(df['text'].str.split().str.len().mean())  # Mean of words
print(df['text'].str.split().str.len().describe())  # Compute more statics

print("\nCounts of each class to see if we need to balance")
print(df['class'].value_counts())

print("\nSearch for any null value")
print(df.isnull())  # See each row result
print(df.isnull().values.any())  # One unique result
print(df.isnull().sum())  # One result per class
print(df.isnull().sum().sum())  # Sum results of each class


# Training a model
train, test = train_test_split(df, test_size=0.33, random_state=42)
# print(train.head())  # Show firsts two
# print(test.head())
print("\nData is still balanced")
print("Train length: ", len(train))
print("Test length:", len(test))
print("\nTrain value counts: ", train['class'].value_counts())
print("Test value counts: ", test['class'].value_counts())

train_x = train['text'].to_list()
train_y = train['class'].to_list()
test_x = test['text'].to_list()
test_y = test['class'].to_list()

# Bag of words model -> Language processing method to represent text as a bag of words
vectorizer = CountVectorizer()
# Fit_transform & transform ->
#   Fit is used on the training data so that we can scale and also learn the scaling parameters of that data.
# The fit method is calculating the mean and variance of each of the features present in our data.
# The transform method is transforming all the features using the respective mean and variance.
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)  # Use the same mean and variance calculated from our training data.

print("\nWe have ", len(train_x), "opinions")  # 402 opinions in out list
print("We have ", train_x_vectors.shape[0], "opinions and ", train_x_vectors.shape[1], "words")
print("Now we have an opinion X word matrix with the number of occurrences for each opinion")
print(train_x_vectors)

# Model creation
clf_dec = DecisionTreeClassifier()
# Fit -> Adjust de Decision Tree with the data where X are the data transformation and y are the output categories
clf_dec.fit(train_x_vectors, train_y)
clf_svm = SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

# Model evaluation
print("Model prediction using decision tree for first test opinion: ", clf_dec.predict(test_x_vectors[0]))
print("\nModel accuracy using decision tree: ", clf_dec.score(test_x_vectors, test_y))
print("Model F1 score (Score class 1, Score class 2) with decision tree", f1_score(test_y, clf_dec.predict(test_x_vectors), average=None))
print("Model confusion matrix: ((TP, FP) (FN, TN)) with decision tree\n", confusion_matrix(test_y, clf_dec.predict(test_x_vectors)))

print("Model prediction using SVC: ", clf_svm.predict(test_x_vectors[0]))
print("\nModel accuracy using SVC: ", clf_svm.score(test_x_vectors, test_y))
print("Model F1 score (Score class 1, Score class 2) with SVC", f1_score(test_y, clf_svm.predict(test_x_vectors), average = None))
print("Model confusion matrix: ((TP, FP) (FN, TN)) with SVC\n", confusion_matrix(test_y, clf_svm.predict(test_x_vectors)))

new_opinions = ['I like my new Leica camera', 'My Tesla car is not working', 'This Nikon camera is useful']
print("\nNew created opinions:", new_opinions)
print("New opinions test with decision tree: ", clf_dec.predict(vectorizer.transform(new_opinions)))
print("New opinions test with SVC: ", clf_svm.predict(vectorizer.transform(new_opinions)))
