import pickle

import numpy as np
from sklearn.datasets import load_files
import joblib
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

DATA_DIR = "dataset/dataset_5classes/dataset_5classes"

data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace")
metadata = np.unique(data.target, return_counts=True)

labels = metadata[0]
count = metadata[1]
class_names = data.target_names

print("cls/name/no_of_docs")
for i in range(len(labels)):
    print(labels[i], class_names[i], count[i])

print("-----------------------------------------------------------------")
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20,
                                                    random_state=False)  # test_size=0.50
print("X_test.shape=", len(X_test))
print("Y_test.shape=", y_test.shape)
print("X_train.shape=", len(X_train))
print("y_train.shape=", y_train.shape)
print("-----------------------------------------------------------------")

# count vectorizer for BOW, just implemented and not used
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
# tfidf -used here
vectorizer = TfidfVectorizer(stop_words="english", max_features=2000, decode_error="ignore")
vectorizer.fit(X_train)
# keywords = vectorizer.get_feature_names()
X_train_vectorized = vectorizer.transform(X_train)

with open("model/classes.pkl", "wb") as fptr:
    pickle.dump(class_names, fptr)


svm = SVC(kernel= 'linear', random_state=1, C=0.1, probability=True)
svm.fit(vectorizer.transform(X_train), y_train)

joblib.dump(svm, 'model/nb_model.h5')
joblib.dump(vectorizer, 'model/vectorizer')

y_pred = svm.predict(vectorizer.transform(X_test))
print("accuracy with svm = ", accuracy_score(y_test, y_pred))
