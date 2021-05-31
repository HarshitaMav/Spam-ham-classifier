#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import seaborn as sns
from plotly import graph_objs as go
import matplotlib as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image


# In[2]:


message_data = pd.read_csv("spam.csv",encoding = "latin")
message_data.head()


# In[3]:


message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)


# In[4]:


message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})


# In[5]:


message_data.info()


# In[6]:


message_data.groupby('Spam/Not_Spam').describe()


# In[7]:


message_data_copy = message_data['message'].copy()


# In[8]:


def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)


# In[9]:


message_data_copy = message_data_copy.apply(text_preprocess)


# In[10]:


message_data_copy


# In[11]:


vectorizer = TfidfVectorizer("english")


# In[12]:


message_mat = vectorizer.fit_transform(message_data_copy)
message_mat


# In[13]:


message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat, 
message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)


# In[15]:


def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words


# In[16]:


message_data_copy = message_data_copy.apply(stemmer)
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)


# In[17]:


message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat, 
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)


# In[19]:


message_data['length'] = message_data['message'].apply(len)
message_data.head()


# In[20]:


length = message_data['length'].to_numpy()
new_mat = np.hstack((message_mat.todense(),length[:, None]))


# In[21]:


message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(new_mat, 
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)


# In[23]:


import matplotlib.pyplot as plt
import csv

x = ['spam','ham']
y = []

with open('spam.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    s= 0
    h = 0

    for row in plots:
        if row[0] == 'spam':
            s += 1
        elif row[0] == 'ham':
            h += 1
    y = [s,h]
plt.bar(x, y, color='g', width=0.72, label="Mails")
plt.xlabel('Label')
plt.ylabel('Mails')
plt.legend()
plt.show()


# In[24]:


plt.pie(y,labels =x,autopct='%1.0f%%')
plt.title('Classified Mails')
plt.show()


# In[25]:


plt.figure(figsize=(12, 8))

message_data[message_data['Spam/Not_Spam']=='ham'].length.plot(bins=35, kind='hist', color='blue', 
                                       label='Ham messages', alpha=0.6)
message_data[message_data['Spam/Not_Spam']=='spam'].length.plot(kind='hist', color='red', 
                                       label='Spam messages', alpha=0.6)
plt.legend()
plt.xlabel("Message Length")


# In[26]:


ham_df = message_data[message_data['Spam/Not_Spam'] == 'ham']['length'].value_counts().sort_index()
spam_df = message_data[message_data['Spam/Not_Spam'] == 'spam']['length'].value_counts().sort_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ham_df.index,
    y=ham_df.values,
    name='ham',
    fill='tozeroy',
))
fig.add_trace(go.Scatter(
    x=spam_df.index,
    y=spam_df.values,
    name='spam',
    fill='tozeroy',
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Ham/Spam Classified</span>'
)
fig.update_xaxes(range=[0, 70])
fig.show()


# In[27]:


c_mask = np.array(Image.open('comment.png'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=c_mask,
)
wc.generate(' '.join(text for text in message_data.loc[message_data['Spam/Not_Spam'] == 'ham', 'message']))
plt.figure(figsize=(18,10))
plt.title('Top words for HAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[28]:


c_mask = np.array(Image.open('comment.png'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=c_mask,
)
wc.generate(' '.join(text for text in message_data.loc[message_data['Spam/Not_Spam'] == 'spam', 'message']))
plt.figure(figsize=(18,10))
plt.title('Top words for SPAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()

