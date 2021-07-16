import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

messages=pd.read_csv(r"/Users/rushabh/Downloads/spam.csv", encoding = "ISO-8859-1")
messages=messages.iloc[:,0:2]

messages.rename(columns={'v1':'label','v2':'messages'},inplace=True)
messages.head()

ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub("[^azA-Z]",' ',str(messages['messages'][i]))
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english') ]
    review=" ".join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,:1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(X_train,y_train)
y_pred=MNB.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_n=confusion_matrix(y_pred,y_test)
print(confusion_n)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
