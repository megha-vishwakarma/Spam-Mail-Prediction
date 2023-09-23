#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,classification_report


# In[2]:


df=pd.read_csv(r"C:\Users\vmegh\Desktop\SIP\spam.csv",encoding='cp1252')
df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.isnull().sum()


# In[7]:


#handle duplicates
df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.shape


# In[10]:


# Check the column names in your DataFrame
print(df.columns)


# In[11]:


df.rename(columns={"v1": "Target","v2": "Text"}, inplace = True)
df


# In[12]:


df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1,inplace = True)


# In[13]:


df.isnull().sum()


# In[23]:


# Assuming 'Target' is the column containing categories
sns.countplot(data=df, x='Target')  # Replace 'df' with your DataFrame and 'Target' with your column name
plt.title('Distribution of Categories')
plt.show()


# In[24]:


plt.pie(df['Target'].value_counts(),autopct='%0.2f',labels=['ham','spam'])
plt.show()


# In[26]:


LE = LabelEncoder()

df['Target'] = LE.fit_transform(df['Target'])
df['Target'].value_counts()


# In[31]:


import nltk
nltk.download('punkt')

# Total No. of Characters in Data
df["characters"] = df["Text"].apply(len)
# Total No. of Words in Data
df["word"] = df["Text"].apply(lambda x:len( nltk.word_tokenize(x)))
# Total No. of Sentence
df["sentence"] = df["Text"].apply(lambda x:len(nltk.sent_tokenize(x)))
df


# In[34]:


sns.pairplot(df,hue="Target")


# In[38]:


cmap = 'Blues'
# Create the heatmap with the specified colormap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(df.corr(), annot=True, cmap=cmap)
# Show the plot
plt.show()


# In[39]:


# Lowercase the text and remove punctuation.
df['Text'] = df['Text'].str.lower().str.replace('[^\w\s]', '')

# Download stopwords from NLTK and remove them from the text.
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['new_input'] = df['Text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))


# In[40]:


wc = WordCloud(
    background_color=None,
    width=800,
    height=400
)


# Wordcloud for SPAM
spam_wc = wc.generate(df[df["Target"] ==1]["Text"].str.cat(sep=" "))

# Wordcloud for HAM
ham_wc = wc.generate(df[df["Target"] ==0]["Text"].str.cat(sep=" "))
# SPAM 
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation="bilinear")
plt.axis("off")
plt.show()
# SPAM 
plt.figure(figsize=(10, 5))
plt.imshow(ham_wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[41]:


# Wordcloud for SPAM
spam_wc_1 = wc.generate(df[df["Target"] ==1]["Text"].str.cat(sep=" "))

# Wordcloud for HAM
ham_wc_1 = wc.generate(df[df["Target"] ==0]["Text"].str.cat(sep=" "))
# SPAM 
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc_1, interpolation="bilinear")
plt.axis("off")
plt.show()
# SPAM 
plt.figure(figsize=(10, 5))
plt.imshow(ham_wc_1, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[58]:


x = df['Text']
y = df['Target']

xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.2, random_state=42)


# In[59]:


tfidf_vectorizer = TfidfVectorizer()
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xtest_tfidf = tfidf_vectorizer.transform(xtest)
print(xtrain_tfidf)
print(xtest_tfidf)


# In[72]:


# Models that are going to be trained
models={
    "Gaussian NB" : GaussianNB(),
    "Multinomial NB" : MultinomialNB(),
    "Bernoulli NB" : BernoulliNB(),
    "Logistic Regression" : LogisticRegression(),
    "SVC" : SVC(),
    "Decision Tree" : DecisionTreeClassifier(),
    "KNN" : KNeighborsClassifier(),
    "Bagging CLF" : BaggingClassifier(),
    "Random Forest" : RandomForestClassifier(),
    "ETC" : ExtraTreesClassifier(),
    "Ada Boost" : AdaBoostClassifier(),
    "Gradient Boost" : GradientBoostingClassifier(),
    "XGB" : XGBClassifier(),
}


# In[73]:


# Initilizing TFIDF Vectorizer
tfidv = TfidfVectorizer(max_features=3000)
# Independent Feature
X = tfidv.fit_transform(df["new_input"]).toarray()
# Dependent Feature
Y = df["Target"].values
# Performing Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
# Creating a function train each model and calculate/return accuracy and precision
def train_clf (model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    return accuracy,precision


# In[74]:


acc_s=[]
pre_s=[]

for name, model in models.items():
    accuracy, precision = train_clf(model, X_train, Y_train, X_test, Y_test)
    
    acc_s.append(accuracy)
    pre_s.append(precision)


# ### Result

# In[75]:


# As Precision matter over Accuracy in this Data, Sorting in DESC order of Precision. All Scores of Models

scores_df = pd.DataFrame({"Algorithm": models.keys(), 
                          "Accuracy": acc_s, 
                         "Precision": pre_s}).sort_values(by="Precision", ascending=False)
scores_df


# In[ ]:





# In[80]:


df.info()


# In[111]:


# loading the data from csv file to a pandas Dataframe
raw_mail_data=pd.read_csv(r"C:\Users\vmegh\Desktop\SIP\spam.csv",encoding='cp1252')


# In[112]:


print(raw_mail_data)


# In[113]:


# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[114]:


# printing the first 5 rows of the dataframe
mail_data.head()


# In[115]:


# checking the number of rows and columns in the dataframe
mail_data.shape


# In[139]:


mail_data.rename(columns={"v1": "Category","v2": "Message"}, inplace = True)
mail_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1,inplace = True)
mail_data.info()


# In[140]:


# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[120]:


df.rename(columns={"v1": "Category","v2": "Message"}, inplace = True)
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1,inplace = True)
df


# In[141]:


# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']


# In[142]:


print(X)


# In[143]:


print(Y)


# #### Splitting the data into training data & test data

# In[144]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[145]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# #### Feature Extraction

# In[146]:


# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[147]:


print(X_train)


# In[148]:


print(X_train_features)


# ### Training the Model
# 
# * Logistic Regression

# In[149]:


model = LogisticRegression()


# In[150]:


# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# * Evaluating the trained model

# In[151]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[152]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[153]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[154]:


print('Accuracy on test data : ', accuracy_on_test_data)


# ### Building a Predictive System

# In[155]:


input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[156]:


input_mail = ["New Mobiles from 2004, MUST GO! Txt: NOKIA to No: 89545 & collect yours today!"]	
# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:




