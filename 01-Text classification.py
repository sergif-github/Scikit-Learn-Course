import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# Import and show dataset
# Dataset -> https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
print("\nReading IMDB Dataset.csv")
df_review = pd.read_csv('IMDB Dataset.csv')
print(df_review)

print("\nSelecting first 9000 and 1000 with positive and negative sentiment respectly")
df_review_positive = df_review[df_review['sentiment'] == 'positive'][:9000]
df_review_negative = df_review[df_review['sentiment'] == 'negative'][:1000]
df_review_reduced_des = pd.concat([df_review_positive, df_review_negative])
print("\nWe have a reduced unbalanced dataset", df_review_reduced_des)
print("Sentiment counts:", df_review.value_counts('sentiment'))


# Dataset reduce
# Oversampling -> Generates new random data
# Undersampling -> Reduce to random existent data
rus = RandomUnderSampler()
df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(df_review_reduced_des[['review']], df_review_reduced_des[['sentiment']])
print("\nWe have a reduced balanced dataset", df_review_bal)
print("Sentiment counts:", df_review_bal.value_counts('sentiment'))


# Create split datasets train and test with fixed random seed
train, test = train_test_split(df_review_bal, test_size=.033, random_state=42)
print("\nTrain dataset", train)
print("\nTest dataset", test)
# Get inputs and outputs of train and test datasets
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']


print("\nWe need to transform text into numbers that our model can understand\n")
print("First option is to use count vectorizer to count the appears of different words")
text = ["This is a positive review.", "This is a negative review."]
df = pd.DataFrame({'review': ['review1', 'review2'], 'text': text})
cv = CountVectorizer()
cv_matrix = cv.fit_transform(df['text'])
df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['review'].values, columns=cv.get_feature_names_out())
print("This are our reviews:")
print(text)
print("This are our reviews word counts:")
print(df_dtm)

print("\nSecond option is to use Tfidf vectorizer to count the appears frequency of different words")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['text'])
df_dtm_freq = pd.DataFrame(tfidf_matrix.toarray(), index=df['review'].values, columns=tfidf.get_feature_names_out())
print("\nThis are our reviews word counts frequency:")
print(df_dtm_freq)



print("\nWe use TfidfVectorizer as option")
tfidf = TfidfVectorizer()
# Fit finds the best parameters. Transform apply this parameters.
# Here we transform train and test to numerical data
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)
# We have milions of values in train_x_vector and test_x_vector stored in two sparse matrx.
# Some words will obtain frequencies of 0.



print("\nNow we will select a model")
print("In supervised learning we know our inputs and our output (review - sentiment).")
print("In non supervised learning we will need to predict patrons in inputs or in output.")
print("In supervised learning we have regression problems (numerical) or classification problems (categories)")
print("We have a classification supervised learning problem")
print("To find the best model we will use support vector machines, decision tree, naive bayes and logistic regression")



# print("\nCreate our models and fit them with train data")
# print("Using support vector machine (SVM)")
svc = SVC(kernel="linear")  # Use different kernels to try...
svc.fit(train_x_vector, train_y)
#print("SVC first prediction:", svc.predict(tfidf.transform(['A good movie'])))
#print("SVC second prediction:", svc.predict(tfidf.transform(['The movie was ugly, bad actors.'])))
#print("SVC third prediction:", svc.predict(tfidf.transform(['I like this movie, it was excellent.'])))

# print("Using decision tree classifier")
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

# print("Using naive bayes")
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# print(Using logistic regression")
lr = LogisticRegression()
lr.fit(train_x_vector, train_y)



print("\nEvaluate our models with test data")
print("Evaluate using accuracy (score):")
print("SVM accuracy:", svc.score(test_x_vector, test_y))
print("Decision tree accuracy:", gnb.score(test_x_vector.toarray(), test_y))
print("Naive bayes accuracy:", dec_tree.score(test_x_vector.toarray(), test_y))
print("Logistic regression accuracy:", lr.score(test_x_vector, test_y))

print("\nEvaluate our model using F1 score ")
print("If we had unbalanced data F1 score wil be better than accuracy")
print("F1 score formula is (2*(recall*precision))/(recall+precision) with min and max value result 0 <-> 1")
print("SVM F1 score: ", f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None))
print("Decision tree F1 score: ", f1_score(test_y, gnb.predict(test_x_vector.toarray()), labels=['positive', 'negative'], average=None))
print("Naive bayes F1 score: ", f1_score(test_y, dec_tree.predict(test_x_vector.toarray()), labels=['positive', 'negative'], average=None))
print("Logistic regression F1 score: ", f1_score(test_y, lr.predict(test_x_vector), labels=['positive', 'negative'], average=None))

print("\nEvaluate our model using classification report")
# Score is the number of data used
# Precision is the proportion of positive that model classified that are actually positives
# Precision focus is to detect false positives
# Recall is the proportion of actually positives that the model classifies as positives
# Recall focus is to detect false negatives
# F1 score is a combination of precision and recall in one metric
print("SVM classification report: \n", classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative']))
print("Decision tree classification report: \n", classification_report(test_y, gnb.predict(test_x_vector.toarray()), labels=['positive', 'negative']))
print("Naive bayes classification report: \n", classification_report(test_y, dec_tree.predict(test_x_vector.toarray()), labels=['positive', 'negative']))
print("Logistic regression classification report: \n", classification_report(test_y, lr.predict(test_x_vector), labels=['positive', 'negative']))




































