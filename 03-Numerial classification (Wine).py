import pandas as pd     # Data into dataframe
import seaborn as sns   # Graphical view
import matplotlib.pyplot as plt   # Graphical view
from sklearn.ensemble import RandomForestClassifier   # Classifier
from sklearn.svm import SVC   # Classifier
from sklearn import svm   # Classifier
from sklearn.neural_network import MLPClassifier   # Classifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score   # Model evaluation
from sklearn.preprocessing import StandardScaler, LabelEncoder   # Preprocessing
from sklearn.model_selection import train_test_split   # Preprocessing

# Problem to solve: Find a classifier to split between good and bad wines
# Load dataset
# Dataset -> https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
wine = pd.read_csv('winequality-red.csv', sep=';')
print("First dataset rows:\n", wine.head())  # View head rows, we see 11 features
print("\nDataset info:")
wine.info()  # View dataset info, we see 1599 rows with non-null float values
print("\nHow many null values we have?\n", wine.isnull().sum())

# Preprocessing data
print("\nWe have this quality possible values:", wine['quality'].unique())
bins = (2, 6.5, 8)
group_names = ['bad', 'good']  # Change quality categories to two numbers
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
print("Now we have this possible values:", wine['quality'].unique())
label_quality = LabelEncoder()  # Encode the bad / good values to a 0 or 1
wine['quality'] = label_quality.fit_transform(wine['quality'])
print("\nFirst 10 dataset rows:\n", wine.head(10))  # View head rows, we see 11 features
print("\nQuality value counts: ", wine['quality'].value_counts())
# sns.barplot(x=wine['quality'].value_counts().index, y=wine['quality'].value_counts().values)
# sns.countplot(wine['quality']) Not working?
sns.countplot(x='quality', data=wine)
plt.show()

# Dataset split
X = wine.drop('quality', axis=1)    # inputs
y = wine['quality']                 # outputs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Values standardization, all features must be taken in account
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print("\nMy first train dataset row of scaled values: ", X_train[0])

# Random forest classifier
# Multiple decision trees are created using different random subsets of the data and features.
# Each decision provides its opinion on how to classify the data.
# Predictions are made by calculating the prediction for each decision tree, then taking the most popular result.
rfc = RandomForestClassifier(n_estimators=200)  # 200 trees / models
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print("\nRandom forest classifier report\n", classification_report(y_test, pred_rfc))
print("Random forest classifier confusion matrix: \n", confusion_matrix(y_test, pred_rfc))
print("\nRandom forest classifier accuracy score: \n", accuracy_score(y_test, pred_rfc))

# SVM classifier (fastest and easiest to apply)
clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print("\nSVM classifier report\n", classification_report(y_test, pred_clf))
print("SVM classifier confusion matrix: \n", confusion_matrix(y_test, pred_clf))
print("\nSVM classifier accuracy score: \n", accuracy_score(y_test, pred_clf))

# Neural Network classifier
mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500)
# 11 based on the number of features
# 500 data iterations to train the classifier
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
print("\nNeural Network classifier report\n", classification_report(y_test, pred_mlpc))
print("Neural Network classifier confusion matrix: \n", confusion_matrix(y_test, pred_mlpc))
print("\nNeural Network classifier accuracy score: \n", accuracy_score(y_test, pred_mlpc))


