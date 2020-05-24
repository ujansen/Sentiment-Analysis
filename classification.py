# If you're using the .csv file I uploaded, execute line 2, otherwise go ahead normally.
new_dataset = pd.read_csv('Cleaned dataset.csv')

"""Splitting the dataset (If you executed line 2, keep the values as they are. If you did not execute line 2, 
change the 1 for X_train and X_test to 0, and change the 2 for y_train and y_test to 1)"""
X_train = new_dataset.iloc[:40000, 1].values
X_test = new_dataset.iloc[40000:, 1].values
y_train = new_dataset.iloc[:40000, 2].values
y_test = new_dataset.iloc[40000:, 2].values
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary = False, ngram_range = (1,3))
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

"""Label Encoding the dependent variable 
Since our dataset has only two possible values for sentiment, we can use label encoder without having to worry too much. 
However, if multiple possibilites were present, we should have used one hot encoding or multilabel classifier"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Training our model using Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 500, random_state = 0)
lr.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
lr_accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train,cv = 10)
print(lr_accuracies.mean())
print(lr_accuracies.std())

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1.0, 0.1, 0.01], 'max_iter': [500, 700, 1000]}]
grid_search = GridSearchCV(estimator = lr, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
lr_best_accuracy = grid_search.best_score_
lr_best_parameters = grid_search.best_params_

# Prediction using our Logistic Regression model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
lr_predictions = lr.predict(X_test)
print('Accuracy of Logistic Regression is: ', accuracy_score(y_test, lr_predictions) * 100)
print(classification_report(y_test,lr_predictions))

# Plotting the confusion matrix - Logistic Regression
import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(confusion_matrix(y_test, lr_predictions), annot = True, fmt = ".0f",
            linewidths = .5, square = True, cmap = 'YlGnBu')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, lr_predictions) * 100)
plt.title(all_sample_title, size = 15)

# Training our model using Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
mnb_accuracies = cross_val_score(estimator = mnb, X = X_train, y = y_train,cv = 10)
print(mnb_accuracies.mean())
print(mnb_accuracies.std())

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'alpha': [1.0, 0.1, 0.01]}]
grid_search = GridSearchCV(estimator = mnb, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
mnb_best_accuracy = grid_search.best_score_
mnb_best_parameters = grid_search.best_params_

# Prediction using our Multinomial Naive Bayes model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
mnb_predictions = mnb.predict(X_test)
print('Accuracy of Multinomial Naive Bayes is: ', accuracy_score(y_test, mnb_predictions) * 100)
print(classification_report(y_test, mnb_predictions))

# Plotting the confusion matrix - Multinomial Naive Bayes
import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(confusion_matrix(y_test, mnb_predictions), annot = True, fmt = ".0f",
            linewidths = .5, square = True, cmap = 'YlGnBu')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, mnb_predictions) * 100)
plt.title(all_sample_title, size = 15)
