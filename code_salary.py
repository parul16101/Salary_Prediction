#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('C:/Users/User/Desktop/19Nov/21/Heroku/salary.csv')


# In[4]:


df.head(5)


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[8]:


df.experience.unique()


# In[15]:


df['experience'].isnull().sum()


# In[16]:


df['experience'].fillna(0, inplace=True)


# In[17]:


df['experience'].isnull().sum()


# In[24]:


def convert(word):
    dic = {
        'one':1,'two':2,'three':3,'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0:0,'thirteen':13,'fourteen':14,'fifteen':15
    }
    return dic[word]


# In[25]:


df['experience'] = df['experience'].apply(lambda x:convert(x))


# In[28]:


df.isnull().sum()


# In[29]:


df['test_score'].fillna(df['test_score'].mean(), inplace=True)
df['interview_score'].fillna(df['interview_score'].mean(), inplace=True)


# In[31]:


df.head()


# In[34]:


X = df.iloc[:,:3]


# In[37]:


y = df.iloc[:,3]


# In[38]:


import seaborn as sns


# In[39]:


sns.pairplot(df)


# In[41]:


t = df.corr()
t['Salary']


# In[45]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[46]:


regressor.fit(X, y)


# In[48]:


regressor.score(X,y)


# In[56]:


# Saving model to disk
import pickle
#pickle.dump(regressor, open('model.pkl','wb'))

with open('model.pickle', 'wb') as f:
    pickle.dump(regressor,f)


# In[61]:


#model = pickle.load(open('model.pkl','rb'))

#path = r'C:\Users\User\Desktop\19Nov\21\Heroku\model.pickle'
#model = pickle.load(open('model.pkl','rb'))

path = r'C:\Users\User\Desktop\19Nov\21\Heroku\model.pickle'
pickle.dump(regressor, open(path, 'wb'))


# In[65]:


# Saving model to disk
path = r'C:\Users\User\Desktop\19Nov\21\Heroku\model.pkl'
pickle.dump(regressor, open(path,'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[ ]:




