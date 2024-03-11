import os
import re
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens


def processing_emails_data():
    director = "C:\\Users\\gabri\\Desktop\\tema_practica_ML\\date\\lingspam_public"
    with open("result_new.txt", "a") as filee:
        filee.write("text,label\n")
    for root, dir, files in os.walk(director):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            if "part10" not in path:
                if "spm" in file:
                    label = 1
                else:
                    label = 0
                cleaned_text = clean_text(content)
                tokens = tokenize_text(cleaned_text)
                filtered_tokens = remove_stopwords(tokens)
                lemmatized_tokens = lemmatize_tokens(filtered_tokens)
                preprocessed_text = ' '.join(lemmatized_tokens)
                with open("result_new.txt", "a") as filee:
                    filee.write(f"{preprocessed_text}, {label}\n")

# processing_emails_data()


def processing_emails_data_test():
    director = "C:\\Users\\gabri\\Desktop\\tema_practica_ML\\date\\lingspam_public"
    with open("result_new_test.txt", "a") as filee:
        filee.write("text,label\n")
    for root, dir, files in os.walk(director):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            if "part10" in path:
                if "spm" in file:
                    label = 1
                else:
                    label = 0
                cleaned_text = clean_text(content)
                tokens = tokenize_text(cleaned_text)
                filtered_tokens = remove_stopwords(tokens)
                lemmatized_tokens = lemmatize_tokens(filtered_tokens)
                preprocessed_text = ' '.join(lemmatized_tokens)
                with open("result_new_test.txt", "a") as filee:
                    filee.write(f"{preprocessed_text}, {label}\n")


# processing_emails_data_test()


df = pd.read_csv('result_new.txt', delimiter=',')
df_test = pd.read_csv('result_new_test.txt', delimiter=',')

spam_words_count = defaultdict(int)
non_spam_words_count = defaultdict(int)
spam_count = 0
non_spam_count = 0

for index, row in df.iterrows():
    words = str(row['text']).split()
    if row['label'] == 1:
        spam_count += 1
        for word in words:
            spam_words_count[word] += 1
    else:
        non_spam_count += 1
        for word in words:
            non_spam_words_count[word] += 1


spam_prob = spam_count / len(df)
non_spam_prob = non_spam_count / len(df)


values = [spam_prob, non_spam_prob]
labels = ['Spam Probability', 'Non-spam Probability']
bars = plt.bar(labels, values, color=['red', 'green'])
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.title('E-mail distribution')

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2%}', ha='center', va='bottom')
plt.show()

print("ALL DATA")
print(f"spam_prob = {spam_prob} , non_spam_prob = {non_spam_prob}")


def predict_spam_bayes(text_line):
    words = text_line.split()
    p_spam = 1
    p_not_spam = 1
    for word in words:
        p_spam *= (spam_words_count[word] + 1) / (spam_count + 1 * 2)
        p_not_spam *= (non_spam_words_count[word] + 1) / (non_spam_count + 1 * 2)

    return 1 if spam_prob * p_spam > non_spam_prob * p_not_spam else 0


print("---------------------------------")
print("Naive Bayes - our algorithm")

correct_predictions = 0
total_predictions = len(df['label'])
accuracy_list = []

for index, row in df.iterrows():
    prediction = predict_spam_bayes(row['text'])
    if prediction == (row['label']):
        correct_predictions += 1
    accuracy_list.append(int(prediction) == int(row['label']))

accuracy = correct_predictions / total_predictions
print(f'Accuracy on training Bayes Naive: {accuracy}')

values = [accuracy, 1-accuracy]
labels = ['Correct', 'Incorrect']
bars = plt.bar(labels, values, color=['green', 'red'])
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.title('Accuracy on training Bayes Naive')

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2%}', ha='center', va='bottom')
plt.show()

spam_prob_label1 = (spam_count - 1) / (len(df['label']) - 1)
spam_prob_label0 = spam_count / (len(df['label']) - 1)
non_spam_prob_label0 = (non_spam_count - 1) / (len(df['label']) - 1)
non_spam_prob_label1 = non_spam_count / (len(df['label']) - 1)

acc = []
acc.append(accuracy)

#testing

correct_predictions_test = 0
total_predictions_test = len(df_test['label'])
accuracy_list_test = []

for index, row in df_test.iterrows():
    prediction_test = predict_spam_bayes(row['text'])
    if prediction_test == row['label']:
        correct_predictions_test += 1
    accuracy_list_test.append(int(prediction_test) == int(row['label']))

accuracy_test = correct_predictions_test / total_predictions_test
print(f'Accuracy on test set Bayes Naive: {accuracy_test}')

incorrect_percentage = (1 - accuracy_test) * 100
correct_percentage = accuracy_test * 100
plt.figure(figsize=(8, 6))
bars = plt.bar(['Correct', 'Incorrect'], [correct_percentage, incorrect_percentage], color=['green', 'red'])

for bar, percentage in zip(bars, [correct_percentage, incorrect_percentage]):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.5, f'{percentage:.2f}%', fontsize=10)

plt.title('Testing Accuracy for Naive Bayes', fontsize=16)
plt.ylabel('Percentage', fontsize=12)
plt.show()


#CVLOO
def predict_spam_CVLOO(text_line, label):
    words = text_line.split()
    p_spam = 1
    p_not_spam = 1

    if label == 1:
        for word in words:
            p_spam *= (spam_words_count[word] - 1 + 1) / (spam_count - 1 + 2)
            p_not_spam *= (non_spam_words_count[word] + 1) / (non_spam_count + 2)
        return 1 if spam_prob_label1 * p_spam > non_spam_prob_label1 * p_not_spam else 0
    else:
        for word in words:
            p_spam *= (spam_words_count[word]+1) / (spam_count+2)
            p_not_spam *= (non_spam_words_count[word] - 1 + 1) / (non_spam_count - 1+2)
        return 1 if spam_prob_label0 * p_spam > non_spam_prob_label0 * p_not_spam else 0

correct_predictions = 0
total_predictions = len(df['label'])

for index, row in df.iterrows():
    prediction = predict_spam_CVLOO(row['text'], int(row['label']))
    if int(prediction) == int(row['label']):
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print(f'Accuracy  - Naive Bayes CVLOO: {accuracy}')

def predict_spam_CVLOO_without_laplace(text_line, label):
    words = text_line.split()
    p_spam = 1
    p_not_spam = 1

    if label == 1:
        for word in words:
            p_spam *= (spam_words_count[word] - 1) / (spam_count - 1)
            p_not_spam *= non_spam_words_count[word] / non_spam_count
        return 1 if spam_prob_label1 * p_spam > non_spam_prob_label1 * p_not_spam else 0
    else:
        for word in words:
            p_spam *= spam_words_count[word] / spam_count
            p_not_spam *= (non_spam_words_count[word] - 1) / (non_spam_count - 1)
        return 1 if spam_prob_label0 * p_spam > non_spam_prob_label0 * p_not_spam else 0


correct_predictions = 0
total_predictions = len(df['label'])

for index, row in df.iterrows():
    prediction = predict_spam_CVLOO_without_laplace(row['text'], int(row['label']))
    if int(prediction) == int(row['label']):
        correct_predictions += 1

accuracy_without_Laplace = correct_predictions / total_predictions
print(f'Accuracy  - Naive Bayes CVLOO without Laplace: {accuracy_without_Laplace}')


def perform_CVLOO(df):
    correct_predictions = 0
    incorrect_predictions = 0

    for index, row in df.iterrows():
        prediction = predict_spam_CVLOO(row['text'], int(row['label']))
        if int(prediction) == int(row['label']):
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    total_predictions = correct_predictions + incorrect_predictions
    accuracy = correct_predictions / total_predictions
    incorrect_percentage = (incorrect_predictions / total_predictions) * 100

    return accuracy, incorrect_percentage


accuracy_bayes_CVLOO, incorrect_percentage_bayes_CVLOO = perform_CVLOO(df)
correct_percentage = accuracy_bayes_CVLOO * 100
plt.figure(figsize=(8, 6))
bars = plt.bar(['Correct', 'Incorrect'], [correct_percentage, incorrect_percentage_bayes_CVLOO], color=['green', 'red'])

for bar, percentage in zip(bars, [correct_percentage, incorrect_percentage_bayes_CVLOO]):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.5, f'{percentage:.2f}%', fontsize=10)

plt.title('Leave-One-Out Cross-Validation Results for Naive Bayes', fontsize=16)
plt.ylabel('Percentage', fontsize=12)
plt.show()

print("---------------------------------")
print("Naive Bayes - using library")

X_train = df['text']
y_train = df['label']

X_test = df_test['text']
y_test = df_test['label']

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)
train_predictions = clf.predict(X_train_vectorized)

print("-------------------------------")
train_accuracy = metrics.accuracy_score(y_train, train_predictions)
print(f'Accuracy on training data using scikit-learn Naive Bayes: {train_accuracy:.2f}')

test_predictions = clf.predict(X_test_vectorized)

test_accuracy = metrics.accuracy_score(y_test, test_predictions)
print(f'Accuracy on testing data using scikit-learn Naive Bayes: {test_accuracy:.2f}')

cvloo_scores = cross_val_score(clf, X_train_vectorized, y_train, cv=5, scoring='accuracy')

print("Naive Bayes using scikit-learn - CVLOO Score:", cvloo_scores.mean())


print("-------------------------------")
print("ID3")
print("-------------------------------")

train_data, test_data, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, train_labels)
predictions = dt_classifier.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)

print("Accuracy on training set:", accuracy)
acc.append(accuracy)

#testing

X_test = vectorizer.transform(df_test['text'])
predictions_test = dt_classifier.predict(X_test)
accuracy_test_id3 = accuracy_score(df_test['label'], predictions_test)

print("Accuracy on Test Set:", accuracy_test_id3)

cv_scores_id3 = cross_val_score(dt_classifier, X_train, train_labels, cv=5)

print("ID3 Mean LOOCV Score:", cv_scores_id3.mean())
print("-------------------------------")
print("K-NN")
print("-------------------------------")

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df['text'])
X_test = vectorizer.transform(df_test['text'])

y_train = df['label']
y_test = df_test['label']
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
predictions_train = knn_classifier.predict(X_train)
predictions_test = knn_classifier.predict(X_test)
accuracy_train = accuracy_score(y_train, predictions_train)
accuracy_test_knn = accuracy_score(y_test, predictions_test)

print("Accuracy on Training Set:", accuracy_train)
acc.append(accuracy_train)
print("Accuracy on Test Set:", accuracy_test_knn)

cv_scores_knn = cross_val_score(knn_classifier, X_train, y_train, cv=5)

print("K-NN Mean LOOCV Score:", cv_scores_knn.mean())

timp_inceput = time.time()

print("-------------------------------")
print("AdaBoost")
print("-------------------------------")

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df['text'])
X_test = vectorizer.transform(df_test['text'])

y_train = df['label']
y_test = df_test['label']

base_classifier = DecisionTreeClassifier(max_depth=1)
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=5, random_state=42)
adaboost_classifier.fit(X_train, y_train)
predictions_train = adaboost_classifier.predict(X_train)
error_train = 1 - adaboost_classifier.score(X_train, y_train)
predictions_test = adaboost_classifier.predict(X_test)
accuracy_test_adaboost = accuracy_score(y_test, predictions_test)

print("Accuracy on Training Set:", 1 - error_train)
acc.append(1 - error_train)
print("Accuracy on Test Set:", accuracy_test_adaboost)

timp_sfarsit = time.time()
timp_total = timp_sfarsit - timp_inceput

print(f"Timpul total de execuție: {timp_total} secunde")

X = vectorizer.fit_transform(df['text'])
y = df['label']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_adaboost = cross_val_score(adaboost_classifier, X, y, cv=cv)

print("Mean Cross-Validation Score:", cv_scores_adaboost.mean())

# grafics for comparison between clasifiers
classifiers = ['Naive Bayes', 'ID3', 'K-NN', 'AdaBoost']
accuracy_values = [accuracy_bayes_CVLOO, np.mean(cv_scores_id3), np.mean(cv_scores_knn), np.mean(cv_scores_adaboost)]
accuracy_percentages = [round(acc * 100, 2) for acc in accuracy_values]

plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, accuracy_values, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.title('Classifier Comparison in Leave-One-Out Cross-Validation')
for bar, value in zip(bars, accuracy_percentages):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value}%', ha='center', va='bottom')
plt.show()

testing_accuracies = [accuracy_test, accuracy_test_id3, accuracy_test_knn, accuracy_test_adaboost]
accuracy_percentages_test = [round(acc * 100, 2) for acc in testing_accuracies]

plt.figure(figsize=(10, 6))
bars_test = plt.bar(classifiers, testing_accuracies, color=['blue', 'green', 'orange', 'red'])

for bar, percentage in zip(bars_test, accuracy_percentages_test):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{percentage}%', fontsize=10)

plt.title('Classifier Comparison - Testing Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)
plt.show()

values = acc
labels = ['Bayes Naiv', 'ID3', 'K-NN', 'AdaBoost']

bars = plt.bar(labels, values, color=['blue', 'green', 'yellow', 'red'])
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.title('Accuracy on training Set: ')
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2%}', ha='center', va='bottom')
plt.show()

print("-------------------------------------------------------------")
testing_accuracies = [accuracy_test, accuracy_test_id3]
classifiers = ['Naive Bayes', 'ID3']
accuracy_percentages_test = [round(acc * 100, 2) for acc in testing_accuracies]
plt.figure(figsize=(8, 6))
bars_test = plt.bar(classifiers, testing_accuracies, color=['blue', 'green'])

for bar, percentage in zip(bars_test, accuracy_percentages_test):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{percentage}%', fontsize=10)
plt.title('Testing Accuracy Comparison - Naive Bayes vs. ID3', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)
plt.show()

testing_accuracies = [accuracy_test, accuracy_test_knn]
classifiers = ['Naive Bayes', 'K-nn']
accuracy_percentages_test = [round(acc * 100, 2) for acc in testing_accuracies]
plt.figure(figsize=(8, 6))
bars_test = plt.bar(classifiers, testing_accuracies, color=['blue', 'green'])

for bar, percentage in zip(bars_test, accuracy_percentages_test):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{percentage}%', fontsize=10)
plt.title('Testing Accuracy Comparison - Naive Bayes vs. K-nn', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)
plt.show()


testing_accuracies = [accuracy_test, accuracy_test_adaboost]
classifiers = ['Naive Bayes', 'AdaBoost']
accuracy_percentages_test = [round(acc * 100, 2) for acc in testing_accuracies]
plt.figure(figsize=(8, 6))
bars_test = plt.bar(classifiers, testing_accuracies, color=['blue', 'green'])
for bar, percentage in zip(bars_test, accuracy_percentages_test):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{percentage}%', fontsize=10)
plt.title('Testing Accuracy Comparison - Naive Bayes vs. AdaBoost', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)
plt.show()