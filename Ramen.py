#!/usr/bin/env python
# coding: utf-8

# In[527]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[528]:


train = pd.read_csv('ramen-ratings.csv')


# In[529]:


train.head()


# In[530]:


train.drop('TopTen',axis=1,inplace=True)
train.drop('Brand',axis=1,inplace=True)
train.drop('Variety',axis=1,inplace=True)


# In[531]:


train.head()


# In[532]:


train.groupby('Style')['Stars'].count()


# In[533]:


def fillStyle(style):
    style = style[0]
    print(style)
    if style == "Cup":
        return 0
    elif style == "Bar":
        return 1
    elif style == "Bowl":
        return 2
    elif style == "Box":
        return 3
    elif style == "Can":
        return 4
    elif style == "Pack":
        return 5
    else:
        return 6


# In[534]:


train['Style']=train[['Style']].apply(fillStyle,axis=1)


# In[535]:


train.head()


# In[536]:


#train.groupby('Country')['Stars'].count()


# In[537]:


def fillCountry(country):
    country = country[0]
    if country=='Japan':
        return 0
    elif country=='USA':
        return 1
    elif country=='South Korea':
        return 2
    elif country=='Taiwan':
        return 3
    elif country=='Thailand':
        return 4
    elif country=='China':
        return 5
    else:
        return 6
train['Country']=train[['Country']].apply(fillCountry,axis=1)


# In[538]:


train.head()


# In[539]:


def fillStars(stars):
    stars = stars[0]
    if  "3" in stars:
        return 1
    if  "2" in stars:
        return 0
    if  "1" in stars:
        return 0
    else:
        return 1


# In[540]:


train['Stars']=train[['Stars']].apply(fillStars,axis=1)


# In[541]:


train.head()


# In[542]:


from sklearn.model_selection import train_test_split


# In[543]:


dummieStars = pd.get_dummies(train['Stars'],drop_first=False)


# In[544]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Stars',axis=1),
                                                    train['Stars'],test_size=0.99,
                                                    random_state=101)
                                                    


# In[545]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(logmodel.score(X_train,y_train))


# In[546]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[560]:





# In[ ]:




