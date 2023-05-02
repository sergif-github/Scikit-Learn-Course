print("Types of learning")
print("Supervised learning:      Making predictions using dada. Input labeled data to predict labels of new data")
print("Unsupervised learning:    Extracting structure from data. Input no labeled data to learn patrons or structures.")
print("                          (Clustering, dimensionality reduction, data association)")
print("Reinforcement learning:   Input states and actions to learn how to improve on best states and actions")

print("\nTerminology")
print("Observation       ->      Each row of a dataset. Also known as sample, example, instance or record...")
print("Feature           ->      Each column of a dataset. Also known as predictor, attribute, input or ind. variable")
print("Response          ->      Each value we are predicting. Also known as target, outcome, label or dep. variable")
print("Classification    ->      In supervised learning models, predicting categorical responses")
print("Regression        ->      In supervised learning models, predicting continuous responses")
print("Clustering        ->      Grouping similar objects into sets (Customer segmentation)")
print("Pre-processing    ->      Feature extraction and normalization (Transform input data)")
print("Dim. reduction    ->      In pre-processing, reducing the number of random variables to consider")
print("Model selection   ->      Comparing, validating and choosing parameters and models (Improve model accuracy)")

print("\nTrain / test split")
print("Train / test split is useful because of its flexibility and speed. However, the chosen observations"
      "\n     can provide a high-variances estimate of out-of-sample accuracy. One way to avoid it is K-fold."
      "\n     K-fold cross-validation is useful when we have a limited dataset or in situations where the dataset is "
      "\n     too small to be divided into separate training and testing sets. By using cross-validation, we can use "
      "\n     all of the data for both training and testing, which can lead to more accurate performance metrics."
      "\n     It is also helpful for assessing how well a model generalizes to new data."
      "\n     It involves dividing the dataset into k equally sized folds and then using k-1 folds for training and "
      "\n     the remaining fold for testing. This process is repeated k times, with each fold being used once as "
      "\n     the test set. The final performance is calculated as the average of the performance of all k folds.")

print("\nAbout scikit-learn")
print("Scikit-learn is build on NumPy, SciPy and matplotlib. It's simple and efficient tool for data mining and "
      "data analysis.\n     Open source with very few restrictions (BSD licence)")
print("Scikit-learn expects features and responses as separate objects.")
print("Scikit-learn expects features and responses to be numerical.")
print("Scikit-learn expects features and responses into numpy arrays.")
print("Scikit-learn expects features and responses to have specific shapes.\n"
      "     First dimension, represented by rows, it's the number of observations."
      "\n     Second dimension, represented by columns, it's the number of features.")

print("\nScikit-learn classification models")
print("First of all: It is possible to adapt regression tasks to classification tasks by transforming the output "
      "variable into\n     a categorical variable with two or more classes and vice-versa.")
print("Linear regression      ->    Supervised learning method used for both classification and regression tasks.\n "
      "    Uses a linear approach to modeling the relationship between the input features and the output variable.")
print("One common approach to using linear regression for classification is logistic regression, which uses a logistic "
      "function to\n     map the continuous output variable of linear regression to a binary output variable(0 or 1)")
print("Logistic regression    ->    Supervised learning method only for regression that uses a logistic function "
      "to estimate\n     the probabilities of two different classes. The output is a probability value between 0 and 1."
      "\n     Can also be multi-class classification using techniques such as one-vs-rest or softmax regression.")
print("Softmax regression     ->    It's an extension of logistic regression, which is used for binary classification."
      "\n     Calculates the probability of membership in each of the classes using a softmax activation function,\n "
      "    which produces a probability distribution over the possible classes.")
print("Decision trees         ->    Supervised learning method used for both classification and regression tasks."
      "\n     The decision tree is built by splitting the data into smaller subsets using the most discriminative "
      "features. \n     The split is chosen based on maximizing the information gain or minimizing the impurity "
      "of the subsets.")
print("Random forest          ->    Supervised learning method used for both classification and regression tasks."
      "\n     Constructs multiple decision trees based on different subsets of the dataset and outputs the most "
      "popular,\n     or average in regression, prediction of all the individual trees.")
print("SVM                    ->    Supervised learning method used for both classification and regression tasks."
      "\n     Support vector machines are a supervised learning method that constructs one or a set of hyperplanes"
      "\n     in a high-dimensional space to separate the data points into different classes.")
print("KNN                    ->    Supervised learning method used for classification and regression. "
      "\n     It predicts based on the class labels of the k-nearest neighbors in the training set."
      "\n     In general, the accuracy of a KNN model when K value increases will follow a Gaussian distribution .")
print("Naive Bayes            ->    Naive Bayes is a supervised learning algorithm used for classification tasks."
      "\n     Is a probabilistic classification method based on Bayes' theorem. The algorithm first estimates "
      "\n     the prior probabilities of each class label based on the training data. Then, for each input feature,"
      "\n     the algorithm estimates the conditional probability of the feature given each class label. "
      "\n     Finally, the algorithm calculates the posterior probability of each class label given the input "
      "features, \n     and predicts the class label with the highest posterior probability.")
print("Neural Networks        ->    Neural networks are a supervised learning method that consist of multiple layers "
      "\n     of interconnected neurons and are trained using backpropagation.")

print("\nOthers")
print("Kaggle:        Popular website of machine learning competitions. ")
print("Iris dataset:  Popular dataset that contains measurements for three different species of iris flowers.")
