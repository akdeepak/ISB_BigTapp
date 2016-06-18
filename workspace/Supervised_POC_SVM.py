import numpy as np
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print(clf.predict(X[2:3]))

with open('C:\\Python_Eclipse\\jan04_tweets\\hashtags_tweets_#Tech.txt', 'r') as tweetdocfile:
    data_train = tweetdocfile.read()

vectorizer = TfidfVectorizer(sublinear_tf=True,min_df=0, max_df=2,
                                 stop_words='english')
X_train = vectorizer.fit_transform(data_train.split(','))
    
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
y_train = np.random.randint(12936, size=12936)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

svm =SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
             max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

svm.fit(X_train, y_train)
svm.predict([[X_train, y_train]])

svm.kernel