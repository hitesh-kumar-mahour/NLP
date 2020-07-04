import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize

# load the dataset of SMS messages
df = pd.read_table('SMSSPamCollection', header=None, encoding='utf-8')
# print(df.head)

classes = df[0]
# print(classes.value_counts())

# convert class labels to binary values, 0 = ham and 1 = spam
encoder = LabelEncoder()
y = encoder.fit_transform(classes)

# store the SMS message data
text_messages = df[1].str.lower()

# use regular expressions to replace email addresses, URLs, phone numbers, other numbers

# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')

# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
'phonenumbr')

# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace and change words to lower case
processed = processed.str.strip()

from nltk.corpus import stopwords
# remove stop words from text messages
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: [term for term in word_tokenize(x) if term not in stop_words])

# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x:[ps.stem(term) for term in x])

# processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in word_tokenize(x)))



# create bag-of-words
all_words = []
for message in processed:
    for w in message:
        all_words.append(w)
all_words = nltk.FreqDist(all_words)

# use the 1500 most common words as features
word_features = [word_tuple[0] for word_tuple in all_words.most_common(1500)]

# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = []
    for word in words:
        if word in word_features:
            features.append(word)

    return features

featureset = []
for current_words in processed:
    features = [0]* len(word_features)
    for word in current_words:
        if word in word_features:
            index_value = word_features.index(word)
            features[index_value] += 1
    featureset.append(features)

from sklearn import model_selection # model selection is used over cross_validation as it was deprecated
X_train, X_test, y_train, y_test = model_selection .train_test_split(featureset, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Define models to train
classifiers = [
        ("K Nearest Neighbors", KNeighborsClassifier()),
        ("Decision Tree"      , DecisionTreeClassifier()),
        ("Random Forest"      , RandomForestClassifier()),
        ("Logistic Regression", LogisticRegression()),
        ("SGD Classifier"     , SGDClassifier(max_iter = 100)),
        ("Naive Bayes"        , MultinomialNB()),
        ("SVM Linear"         , SVC(kernel = 'linear'))

        ]

for t in classifiers:
    clf = t[1]
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)*100
    print("{} Accuracy: {}".format(t[0], accuracy))

#  Voting classifier
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = classifiers, voting='hard', weights=None)
voting_clf = voting_clf.fit(X_train,y_train)
accuracy = voting_clf.score(X_test,y_test)*100
print("Accuracy for voting classifier: {}".format( accuracy))
