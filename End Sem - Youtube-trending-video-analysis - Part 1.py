#!/usr/bin/env python
# coding: utf-8

# # Part I: Data Gathering and Preprocessing

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ### Importing scikit-learn classifiers

# In[2]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ### Importing Data

# In[3]:


video = pd.read_csv(r"USvideos.csv", header=0)
video.head(5)


# ### Deleting unused columns and renaming the remaining columns

# In[4]:


new_columns = ['title', 'category_id']
new_video = video[new_columns]
new_video.to_csv("USvideos.csv", index=False)
new_video = pd.read_csv("USvideos.csv", header=0, names=['Title', 'Category_ID'])


# ### Importing JSON file

# In[5]:


category_json = pd.read_json("US_category_id.json")
category_json.head(5)


# ### Creating a list of Dictionaries with ID and Category label mapping

# In[6]:


category_dict = [{'id': item['id'], 'title': item['snippet']['title']} for item in category_json['items']]
category_dict


# ### Creating a DataFrame for the Dictionary

# In[7]:


category_df = pd.DataFrame(category_dict)
categories = category_df.rename(index=str, columns = {"id":"Category_ID","title":"Category"})
categories.head(5)


# # Part II: Training

# ### Splitting 'title' into string of words using CountVectorizer

# In[8]:


vector = CountVectorizer()
counts = vector.fit_transform(new_video['Title'].values)


# ### Using various classification models and targetting 'Category'

# In[9]:


NB_Model = MultinomialNB()
RFC_Model = RandomForestClassifier()
SVC_Model = SVC()
KNC_Model = KNeighborsClassifier()
DTC_Model = DecisionTreeClassifier()


# In[10]:


output = new_video['Category_ID'].values


# In[11]:


NB_Model.fit(counts,output)


# In[12]:


RFC_Model.fit(counts,output)


# In[13]:


SVC_Model.fit(counts,output)


# In[14]:


KNC_Model.fit(counts,output)


# In[15]:


DTC_Model.fit(counts,output)


# ### Checking the accuracy using 90/10 train/test split

# In[16]:


X = counts
Y = output
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1)


# In[17]:


NBtest = MultinomialNB().fit(X_train,Y_train)
nb_predictions = NBtest.predict(X_test)
acc_nb = NBtest.score(X_test, Y_test)
print('The Naive Bayes Algorithm has an accuracy of', acc_nb)


# In[18]:


RFCtest = RandomForestClassifier().fit(X_train,Y_train)
rfc_predictions = RFCtest.predict(X_test)
acc_rfc = RFCtest.score(X_test, Y_test)
print('The Random Forest Algorithm has an accuracy of', acc_rfc)


# In[19]:


SVCtest = SVC().fit(X_train,Y_train)
svc_predictions = SVCtest.predict(X_test)
acc_svc = SVCtest.score(X_test, Y_test)
print('The Support Vector Algorithm has an accuracy of', acc_svc)


# In[20]:


KNCtest = KNeighborsClassifier().fit(X_train,Y_train)
knc_predictions = KNCtest.predict(X_test)
acc_knc = KNCtest.score(X_test, Y_test)
print('The K Neighbors Algorithm has an accuracy of', acc_knc)


# In[21]:


DTCtest = DecisionTreeClassifier().fit(X_train,Y_train)
dtc_predictions = DTCtest.predict(X_test)
acc_dtc = DTCtest.score(X_test, Y_test)
print('The Decision Tree Algorithm has an accuracy of', acc_dtc)


# # Part III: Test

# ### Entering hypothetical titles to predict the category

# In[22]:


Titles = ["Liverpool vs Barcelona football match highlights"]


# ### Inserting above titles into each classifier model

# In[23]:


Titles_counts = vector.transform(Titles)


# ### Naive Bayes Model

# In[24]:


PredictNB = NB_Model.predict(Titles_counts)
PredictNB


# ### Random Forest Model

# In[25]:


PredictRFC = RFC_Model.predict(Titles_counts)
PredictRFC


# ### SVC Model

# In[26]:


PredictSVC = SVC_Model.predict(Titles_counts)
PredictSVC


# ### K Neighbors Model

# In[27]:


PredictKNC = KNC_Model.predict(Titles_counts)
PredictKNC


# ### Decision Tree Model

# In[28]:


PredictDTC = DTC_Model.predict(Titles_counts)
PredictDTC


# ### Output will be an array of numbers. Iterate through the Category Dictionary (from JSON file) to find "title"

# In[29]:


CategoryNamesListNB = []
for Category_ID in PredictNB:
    MatchingCategoriesNB = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesNB:
        CategoryNamesListNB.append(MatchingCategoriesNB[0]["title"])


# In[30]:


CategoryNamesListRFC = []
for Category_ID in PredictRFC:
    MatchingCategoriesRFC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesRFC:
        CategoryNamesListRFC.append(MatchingCategoriesRFC[0]["title"])


# In[31]:


CategoryNamesListSVC = []
for Category_ID in PredictSVC:
    MatchingCategoriesSVC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesSVC:
        CategoryNamesListSVC.append(MatchingCategoriesSVC[0]["title"])


# In[32]:


CategoryNamesListKNC = []
for Category_ID in PredictKNC:
    MatchingCategoriesKNC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesKNC:
        CategoryNamesListKNC.append(MatchingCategoriesKNC[0]["title"])


# In[33]:


CategoryNamesListDTC = []
for Category_ID in PredictDTC:
    MatchingCategoriesDTC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesDTC:
        CategoryNamesListDTC.append(MatchingCategoriesDTC[0]["title"])


# ### Mapping these values to the Titles we want to Predict

# In[34]:


TitleDataFrameNB = []
for i in range(0, len(Titles)):
    TitleToCategoriesNB = {'Title': Titles[i],  'Category': CategoryNamesListNB[i]}
    TitleDataFrameNB.append(TitleToCategoriesNB)


# In[35]:


TitleDataFrameRFC = []
for i in range(0, len(Titles)):
    TitleToCategoriesRFC = {'Title': Titles[i],  'Category': CategoryNamesListRFC[i]}
    TitleDataFrameRFC.append(TitleToCategoriesRFC)


# In[36]:


TitleDataFrameSVC = []
for i in range(0, len(Titles)):
    TitleToCategoriesSVC = {'Title': Titles[i],  'Category': CategoryNamesListSVC[i]}
    TitleDataFrameSVC.append(TitleToCategoriesSVC)


# In[37]:


TitleDataFrameKNC = []
for i in range(0, len(Titles)):
    TitleToCategoriesKNC = {'Title': Titles[i],  'Category': CategoryNamesListKNC[i]}
    TitleDataFrameKNC.append(TitleToCategoriesKNC)


# In[38]:


TitleDataFrameDTC = []
for i in range(0, len(Titles)):
    TitleToCategoriesDTC = {'Title': Titles[i],  'Category': CategoryNamesListDTC[i]}
    TitleDataFrameDTC.append(TitleToCategoriesDTC)


# ### Converting the resulting Dictionary to a Data Frame

# In[39]:


PredictDFnb = pd.DataFrame(PredictNB)
TitleDFnb = pd.DataFrame(TitleDataFrameNB)
PreFinalDFnb = pd.concat([PredictDFnb, TitleDFnb], axis=1)
PreFinalDFnb.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFnb = PreFinalDFnb.drop(['Categ_ID'],axis=1)
colsNB = FinalDFnb.columns.tolist()
colsNB = colsNB[-1:] + colsNB[:-1]
FinalDFnb= FinalDFnb[colsNB]


# In[40]:


PredictDFrfc = pd.DataFrame(PredictRFC)
TitleDFrfc = pd.DataFrame(TitleDataFrameRFC)
PreFinalDFrfc = pd.concat([PredictDFrfc, TitleDFrfc], axis=1)
PreFinalDFrfc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFrfc = PreFinalDFrfc.drop(['Categ_ID'],axis=1)
colsRFC = FinalDFrfc.columns.tolist()
colsRFC = colsRFC[-1:] + colsRFC[:-1]
FinalDFrfc= FinalDFrfc[colsRFC]


# In[41]:


PredictDFsvc = pd.DataFrame(PredictSVC)
TitleDFsvc = pd.DataFrame(TitleDataFrameSVC)
PreFinalDFsvc = pd.concat([PredictDFsvc, TitleDFsvc], axis=1)
PreFinalDFsvc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFsvc = PreFinalDFsvc.drop(['Categ_ID'],axis=1)
colsSVC = FinalDFsvc.columns.tolist()
colsSVC = colsSVC[-1:] + colsSVC[:-1]
FinalDFsvc= FinalDFsvc[colsSVC]


# In[42]:


PredictDFknc = pd.DataFrame(PredictKNC)
TitleDFknc = pd.DataFrame(TitleDataFrameKNC)
PreFinalDFknc = pd.concat([PredictDFknc, TitleDFknc], axis=1)
PreFinalDFknc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFknc = PreFinalDFknc.drop(['Categ_ID'],axis=1)
colsKNC = FinalDFknc.columns.tolist()
colsKNC = colsKNC[-1:] + colsKNC[:-1]
FinalDFknc= FinalDFknc[colsKNC]


# In[43]:


PredictDFdtc = pd.DataFrame(PredictDTC)
TitleDFdtc = pd.DataFrame(TitleDataFrameDTC)
PreFinalDFdtc = pd.concat([PredictDFdtc, TitleDFdtc], axis=1)
PreFinalDFdtc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFdtc = PreFinalDFdtc.drop(['Categ_ID'],axis=1)
colsDTC = FinalDFdtc.columns.tolist()
colsDTC = colsDTC[-1:] + colsDTC[:-1]
FinalDFdtc= FinalDFdtc[colsDTC]


# ### Viewing the Final Prediction Results

# # Demo

# In[44]:


Titles = ["Liverpool vs Barcelona football match highlights"]


# In[45]:


Titles_counts = vector.transform(Titles)
PredictDTC = DTC_Model.predict(Titles_counts)

CategoryNamesListDTC = []
for Category_ID in PredictDTC:
    MatchingCategoriesDTC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesDTC:
        CategoryNamesListDTC.append(MatchingCategoriesDTC[0]["title"])

TitleDataFrameDTC = []
for i in range(0, len(Titles)):
    TitleToCategoriesDTC = {'Title': Titles[i],  'Category': CategoryNamesListDTC[i]}
    TitleDataFrameDTC.append(TitleToCategoriesDTC)
    
PredictDFdtc = pd.DataFrame(PredictDTC)
TitleDFdtc = pd.DataFrame(TitleDataFrameDTC)
PreFinalDFdtc = pd.concat([PredictDFdtc, TitleDFdtc], axis=1)
PreFinalDFdtc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFdtc = PreFinalDFdtc.drop(['Categ_ID'],axis=1)
colsDTC = FinalDFdtc.columns.tolist()
colsDTC = colsDTC[-1:] + colsDTC[:-1]
FinalDFdtc= FinalDFdtc[colsDTC]

# Decision Trees
FinalDFdtc


# In[ ]:




