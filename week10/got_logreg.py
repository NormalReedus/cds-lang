# %% Imports

from got_loader import load_got_data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit

import sys
sys.path.append('..')
import utils.classifier_utils as clf



# %% Load training data

# Get training data and labels from custom loader
sentences, seasons = load_got_data()



# %% Training / validation split + vectorization

# Create training / validation split
train_sents, test_sents, train_label, test_label = train_test_split(sentences, 
                                                    seasons, 
                                                    test_size=0.2, 
                                                    random_state=1337)

vectorizer = TfidfVectorizer()

train_sents_feats = vectorizer.fit_transform(train_sents)
test_sents_feats = vectorizer.transform(test_sents)


# %% Create classifier and predict

classifier = LogisticRegression(random_state=1337).fit(train_sents_feats, train_label)

pred_label = classifier.predict(test_sents_feats)


# %% Display metrics

classifier_metrics = classification_report(test_label, pred_label)
print(classifier_metrics)



# %% Cross validation

# Vectorize full dataset
vect_sents = vectorizer.fit_transform(sentences)

# initialise cross-validation method
title = "Learning Curves (Logistic Regression)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=1337)

# run on data
classifier = LogisticRegression(random_state=1337)
clf.plot_learning_curve(classifier, title, vect_sents, seasons, cv=cv, n_jobs=4)

# %%
