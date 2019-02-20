"Final NLP Model, trigrams, no parsing"

"Read in Data"
import pandas as pd
df = pd.read_csv('U:/BMI-733/Data/data.csv',encoding = "ISO-8859-1")
col = ['DC_YEAR', 'scan_field', 'OD']
df = df[col]
df.head()

"Make test and train dataset"
train=df[df['DC_YEAR']==2017]
test=df[df['DC_YEAR']==2018]

X_train=train.scan_field
X_test=test.scan_field
y_train=train.OD
y_test=test.OD

"Parse the data"
import en_core_web_sm

nlp=en_core_web_sm.load()

"To implement this for the entire data, need to create custom tokenizers to use in the countvectorizor"
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer 

def spacy_pos(sentences):
    docs = nlp(sentences)
    pos_text = [tok.lemma_ + tok.pos_ for tok in docs if (tok not in stopwords)]
    return pos_text

count_vect = CountVectorizer(ngram_range=(1,3), stop_words='english', min_df=5)
pos_vect = CountVectorizer(tokenizer=spacy_pos)

"Create word vector to get count in vocabulary"
word_vect = CountVectorizer(ngram_range=(1,1), stop_words='english', min_df=5)

"Create featues for training and test data"
X_train_bow=count_vect.fit_transform(X_train.values.astype('U'))

X_train_pos=pos_vect.fit_transform(X_train.values.astype('U'))

X_train_words=word_vect.fit_transform(X_train.values.astype('U'))

X_test_bow=count_vect.transform(X_test.values.astype('U'))
X_test_pos=pos_vect.transform(X_test.values.astype('U'))

"Combine these sparse matrices to create final feature space"
import scipy as sp
X_train_final=sp.sparse.hstack([X_train_bow, X_train_pos])
X_test_final=sp.sparse.hstack([X_test_bow, X_test_pos])

"plot code"
import itertools
import numpy as np
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

"Count features in the model"
np.shape(X_train_final)
np.shape(X_train_bow)
np.shape(X_train_pos)
np.shape(X_train_words)

"Start training a model"
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold

"Set up repeated cross validation"
cross_validation=RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

"First model"
tuned_parameters = [{'C': [.1, 1, 10, 100]}]

clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=cross_validation, scoring='f1')
clf.fit(X_train_final, y_train)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

y_true, y_pred = y_test, clf.predict(X_test_final)
print(classification_report(y_true, y_pred))

cnf_matrix=metrics.confusion_matrix(y_true, y_pred)
class_names = ['Not OD', 'Drug OD']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

"Already better than the parsing model! Keep tuning"
tuned_parameters = [{'C': [.001, .01, .05, .1, .15]}]

clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=cross_validation, scoring='f1')
clf.fit(X_train_final, y_train)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

y_true, y_pred = y_test, clf.predict(X_test_final)
print(classification_report(y_true, y_pred))

cnf_matrix=metrics.confusion_matrix(y_true, y_pred)
class_names = ['Not OD', 'Drug OD']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

"Another more tuning step"
tuned_parameters = [{'C': [.04, .08, .1, .12]}]

clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=cross_validation, scoring='f1')
clf.fit(X_train_final, y_train)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

y_true, y_pred = y_test, clf.predict(X_test_final)
print(classification_report(y_true, y_pred))

cnf_matrix=metrics.confusion_matrix(y_true, y_pred)
class_names = ['Not OD', 'Drug OD']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

"Looks like 0.12 is going to be the best"
tuned_parameters = [{'C': [.11, .12, .13, .14]}]

clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=cross_validation, scoring='f1')
clf.fit(X_train_final, y_train)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

y_true, y_pred = y_test, clf.predict(X_test_final)
print(classification_report(y_true, y_pred))

cnf_matrix=metrics.confusion_matrix(y_true, y_pred)
class_names = ['Not OD', 'Drug OD']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

"Retrain with C=.11, final model"
clf=SVC(kernel='linear', C=0.11).fit(X_train_final, y_train)

predicted=clf.predict(X_train_final)
print(classification_report(y_train, predicted))

cnf_matrix=metrics.confusion_matrix(y_train, predicted)
class_names = ['Not OD', 'Drug OD']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

predicted_test=clf.predict(X_test_final)
print(classification_report(y_test, predicted_test))

cnf_matrix=metrics.confusion_matrix(y_test, predicted_test)
class_names = ['Not OD', 'Drug OD']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

"Export errors"
export = pd.DataFrame(y_test[y_test != predicted_test], columns=["OD"])
export["predicted"] =predicted_test[y_test != predicted_test]
export=export.join(df, how='inner',lsuffix='_l',rsuffix='_r')


export.to_csv('U:/temp/errors.csv')


"Now save the classifier"
import pickle

with open('C:/Users/pjwa227/Google Drive/BMI-733/Project/New Project/do_classifier_ex_fin', 'wb') as picklefile:
    pickle.dump(clf, picklefile)

"Load model, here for example purposes"
with open('C:/Users/pjwa227/Google Drive/BMI-733/Project/New Project/do_classifier_ex', 'rb') as training_model:  
    model = pickle.load(training_model)






