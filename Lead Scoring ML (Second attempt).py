#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import sklearn.preprocessing as preproc
from sklearn.feature_extraction import text
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_excel(r"C:\Users\cyrine\Downloads\DatabaseNew.xlsx")


# In[3]:


df_dub = df.copy()
df.drop_duplicates(subset={'Lead_Full Name'}, inplace=True)
df.shape


# In[4]:


# Data display coustomization
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[8]:


# I will drop the columns having more than 60% NA values.
df = df.drop(df.loc[:,list(round(100*(df.isnull().sum()/len(df.index)), 2)>60)].columns, 1)


# In[9]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[10]:


#dropping the owner column as they change and the lead name. 

df.drop(['Owner', 'Lead_Full Name'], 1, inplace = True)


# In[11]:


def func(a):
    if "data" in str(a).lower():
        return 1
    else:
        return 0

df["Data"] = df["Job Position"].apply(lambda x: func(x))


# In[12]:


df.head(10)


# In[13]:


def func(a):
    if "marketing" in str(a).lower():
        return 1
    else:
        return 0

df["Marketing"] = df["Job Position"].apply(lambda x: func(x))


# In[14]:


def func(a):
    if "analytics" in str(a).lower() or "analyst" in str(a).lower():
        return 1
    else:
        return 0

df["Analytics"] = df["Job Position"].apply(lambda x: func(x))


# In[15]:


df['Job Position'] = df['Job Position'].replace(np.nan, "Other")


# In[16]:


df.head(5)


# In[17]:


df['Job Position'].value_counts()


# In[18]:


def func(a):
    if "analytics" in str(a).lower() or "analyst" in str(a).lower() or  "data" in str(a).lower() or  "marketing" in str(a).lower()  :
        return 0
    else:
        return 1

df["Other JDs"] = df["Job Position"].apply(lambda x: func(x))


# In[19]:


df.head(5)


# In[20]:


def func(a):
    if "senior" in str(a).lower() or "sr" in str(a).lower():
        return 1
    else:
        return 0

df["Senior"] = df["Job Position"].apply(lambda x: func(x))


# In[21]:


def func(a):
    if "head" in str(a).lower():
        return 1
    else:
        return 0

df["Head"] = df["Job Position"].apply(lambda x: func(x))


# In[22]:


def func(a):
    if "manager" in str(a).lower():
        return 1
    else:
        return 0

df["Manager"] = df["Job Position"].apply(lambda x: func(x))


# In[23]:


df.head(5)


# In[24]:


def func(a):
    if "vp" in str(a).lower():
        return 1
    else:
        return 0

df["VP"] = df["Job Position"].apply(lambda x: func(x))


# In[25]:


def func(a):
    if "director" in str(a).lower():
        return 1
    else:
        return 0

df["Director"] = df["Job Position"].apply(lambda x: func(x))


# In[26]:


def func(a):
    if "manager" in str(a).lower() or "senior" in str(a).lower() or "sr" in str(a).lower() or "vp" in str(a).lower() or "head" in str(a).lower() or "director" in str(a).lower():
        return 0
    else:
        return 1

df["Other Positions"] = df["Job Position"].apply(lambda x: func(x))


# In[27]:


df.head(5)


# In[28]:


df.Segment.describe()


# In[29]:


plt.figure(figsize = (10,5))
ax= sns.countplot(df['Segment'])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 90)
plt.show()


# In[30]:


df['Segment'] = df['Segment'].replace(np.nan, '1')


# In[31]:


df['Segment'] = df['Segment'].replace(['advertiser_cPG', 'CPG','advertiser_CPG', 'Advertiser_CPG', 'advertiser_cpg'], '1')


# In[32]:


df['Segment'] = df['Segment'].replace(['Advertiser_Retail', 'advertiser_Retail','advertiser_retail'], '2')


# In[33]:


df['Segment'] = df['Segment'].replace(['Agency_Media', 'agency_media','media agency', 'Media_Agency'], '3')


# In[34]:


df['Segment'] = df['Segment'].replace(['Advertiser_Finance', 'Advertiser_Fin'], '4')


# In[35]:


df['Segment'] = df['Segment'].replace(['Advertiser_Tech', 'advertiser_tech'], '5')


# In[36]:


df['Segment'] = df['Segment'].replace(['Advertiser_Int'], '6')


# In[37]:


df['Segment'] = df['Segment'].replace(['Agency_Analytics', 'Agency_Anaytics', 'Anaytics_Agency', 'agency_analytics'], '7')


# In[38]:


df['Segment'] = df['Segment'].replace(['Advertiser_Pharma', 'advertiser_pharma'], '8')


# In[39]:


df['Segment'] = df['Segment'].replace(['Advertiser_Entertainment', 'advertiser-entertainment'], '9')


# In[40]:


df['Segment'] = df['Segment'].replace(['Advertiser_Health', 'advertiser_health'], '10')


# In[41]:


df['Segment'] = df['Segment'].replace(['Agency_Consulting', 'Advertiser_Management Consulting', 'Consulting_Agency', 'Agency_consulting', 'Agency Consultancy' ], '11')


# In[42]:


df['Segment'] = df['Segment'].replace(['Advertiser_Telco', 'Advertiser_Auto', 'Advertiser_Tourism & Travel', 'Advertiser_E-com', 'Advertiser_Real Estate', 'Advertiser_Insurance', 'Advertiser_FMCG', 'Advertiser_Education', 'Advertiser_Jewelry', 'Advertiser_Logistics', 'Advertiser_telco', 'Advertiser_Gambling&Casinos', 'Advertiser_Aviation', 'advertiser', 'Advertiser'], '12')


# In[43]:


df['Segment'] = df['Segment'].replace(['Agency', 'Agency_Creative', 'agency'], '13')


# In[44]:


df['Segment'].value_counts()


# In[45]:


df['Continent'].value_counts()


# In[46]:


df['Continent'] = df['Continent'].replace(['america'], 'America')


# In[47]:


df['Continent'].value_counts()


# In[48]:


df['Continent'] = df['Continent'].replace(['europe'], 'Europe')


# In[49]:


df['Continent'] = df['Continent'].replace(['asia'], 'Asia')


# In[50]:


df['Continent'] = df['Continent'].replace(['Connecticut'], 'America')


# In[51]:


df['Continent'].value_counts()


# In[52]:


df['Continent'] = df['Continent'].replace(['America'], '1')


# In[53]:


df['Continent'] = df['Continent'].replace(['Europe'], '2')


# In[54]:


df['Continent'] = df['Continent'].replace(['Asia'], '3')


# In[55]:


df['Continent'] = df['Continent'].replace(['Australia'], '4')


# In[56]:


df['Continent'] = df['Continent'].replace(['Africa'], '5')


# In[57]:


df['Continent'] = df['Continent'].replace(np.nan, '1')


# In[58]:


df['Continent'].value_counts()


# In[59]:


df['Prospection Account'].value_counts()


# In[60]:


df['Prospection Account'] = df['Prospection Account'].replace(np.nan, '1')


# In[61]:


df['Prospection Account'] = df['Prospection Account'].replace(['Ramlas'], '1')


# In[62]:


df['Prospection Account'] = df['Prospection Account'].replace(['personal'], '2')


# In[63]:


df['Prospection Account'] = df['Prospection Account'].replace(['Ramla\'s'], '1')


# In[64]:


df['Prospection Account'] = df['Prospection Account'].replace(['Personal'], '2')


# In[65]:


df['Prospection Account'] = df['Prospection Account'].replace(['ramla\'s'], '1')


# In[66]:


df['Prospection Account'] = df['Prospection Account'].replace(['both'], '3')


# In[67]:


df['Prospection Account'].value_counts()


# In[68]:


df.drop(['Inmail sent date'], 1, inplace = True)


# In[69]:


df['Month followup'].value_counts()


# In[70]:


df['Month followup'] = df['Month followup'].replace([np.nan, '<<<<','-', ' ', '----','`'], 0)


# In[71]:


df['Month followup'] = df['Month followup'].replace([0], '0')


# In[72]:


df['Month followup'].value_counts()


# In[73]:


df['Month Sent Date'] = df['Month Sent Date'].replace(['Ahlem'], '11')


# In[74]:


df['Month Sent Date'] = df['Month Sent Date'].replace([11], '11')


# In[75]:


df['Month Sent Date'] = df['Month Sent Date'].replace(['to send'], '4')


# In[76]:


df['Month Sent Date'] = df['Month Sent Date'].replace([4], '4')


# In[77]:


df['Month Sent Date'] = df['Month Sent Date'].replace([0], '0')


# In[78]:


df['Month Sent Date'] = df['Month Sent Date'].replace(np.nan, '0')


# In[79]:


df['Month Sent Date'].value_counts()


# In[80]:


df['LI_Connection_Status'].value_counts()


# In[81]:


df['LI_Connection_Status'] = df['LI_Connection_Status'].replace(['accepted', 'Accepted', 'Accepted '], 'Yes')


# In[82]:


df['LI_Connection_Status'] = df['LI_Connection_Status'].replace(['accepted'], 'Yes')


# In[83]:


df['LI_Connection_Status'] = df['LI_Connection_Status'].replace(np.nan, 'No')


# In[84]:


df['LI_Connection_Status'] = df['LI_Connection_Status'].replace(['deleted', 'Deleted', 'pending', 'Pending', 'withdrawn', 'Withdrawn'], 'No')


# In[85]:


df['LI_Connection_Status'].value_counts()


# In[86]:


df['State/ Country'] = df['State/ Country'].replace(np.nan, 'New York')


# In[87]:


df.head()


# In[88]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[89]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[90]:


df.isnull().sum()


# In[91]:


# Rest missing values are under 1.5% so we can drop these rows.
df.dropna(inplace = True)


# In[92]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[93]:


df.head()


# In[94]:


df.isnull().sum()


# In[95]:


data_retailed= len(df)* 100 / len(df_dub)
print("{} % of original rows is available for EDA".format(round(data_retailed,2)))


# In[96]:


df.shape


# #Exploratory Data Analytics
# Univariate Analysis 

# In[97]:


df.head()


# In[98]:


print("Original Data {} % Retained".format(round((len(df) * len(df.columns))*100/(len(df_dub.columns)*len(df_dub)),2)))


# In[99]:


varlist = ['LI_Connection_Status']
def binary_map(x):
    return x.map({'Yes': 1, 'No': 0})

# Applying the function to the housing list
df[varlist] = df[varlist].apply(binary_map)
df.head()


# In[100]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(df[['Company Name', 'State/ Country']], drop_first=True)
dummy1.head()


# In[101]:


df = pd.concat([df, dummy1], axis=1)
df.head()


# In[102]:


df = df.drop(['Company Name', 'State/ Country', 'Job Position'], axis = 1)
df.head()


# In[103]:


df.shape


# In[104]:


df.drop(['Invitation Sent Date'], 1, inplace = True)


# In[105]:


df.drop(['Nb'], 1, inplace = True)


# In[106]:


Converted = round((sum(df['LI_Connection_Status'])/len(df['LI_Connection_Status'].index))*100,2)

print("We have almost {} % Acceptance rate".format(Converted))


# In[ ]:





# In[107]:


df.shape


# In[108]:


df.head()


# In[109]:


from sklearn.model_selection import train_test_split
# Putting feature variable to X
X = df.drop(['LI_Connection_Status'], axis=1)


# In[110]:


X.head()


# In[111]:


X.shape


# In[112]:


# Putting response variable to y
y = df['LI_Connection_Status']


# In[113]:


y.head()


# In[114]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)


# In[115]:


X_train.head()


# In[116]:


X_train.shape


# In[117]:


X_test.head()


# In[118]:


X_test.shape


# In[119]:


y_train.head()


# In[120]:


y_train.shape


# In[121]:


y_test.head()


# In[122]:


y_test.shape


# In[123]:


X_list = list(X.columns)


# In[124]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['Prospection Account','Segment','Continent', 'Data','Marketing', 'Analytics', 'Other JDs', 'Senior', 'Head', 'Manager', 'VP', 'Director','Other Positions', 'Month Sent Date']] = scaler.fit_transform(X_train[['Prospection Account','Segment','Continent', 'Data','Marketing', 'Analytics', 'Other JDs', 'Senior', 'Head', 'Manager', 'VP', 'Director','Other Positions', 'Month Sent Date']])
X_test[['Prospection Account','Segment','Continent', 'Data','Marketing', 'Analytics', 'Other JDs', 'Senior', 'Head', 'Manager', 'VP', 'Director','Other Positions', 'Month Sent Date']] = scaler.transform(X_test[['Prospection Account','Segment','Continent', 'Data','Marketing', 'Analytics', 'Other JDs', 'Senior', 'Head', 'Manager', 'VP', 'Director','Other Positions', 'Month Sent Date']])

X_train.head()


# In[125]:


# Checking the Converted Rate
Converted = round((sum(df['LI_Connection_Status'])/len(df['LI_Connection_Status'].index))*100,2)
print("We have almost {} %  Converted rate after successful data manipulation".format(Converted))


# In[126]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier()


# In[127]:


# number of trees used
print('Number of Trees used : ', model.n_estimators)


# In[128]:


# fit the model with the training data
model.fit(X_train,y_train)


# In[129]:


# predict the target on the train dataset
predict_train = model.predict(X_train)
predict_train


# In[130]:


trainaccuracy = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', trainaccuracy)


# # VIF

# In[131]:


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train, predict_train )
print(confusion)


# In[132]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[133]:


# Let's see the sensitivity of our model
trainsensitivity= TP / float(TP+FN)
trainsensitivity


# In[134]:


# Let us calculate specificity
trainspecificity= TN / float(TN+FP)
trainspecificity


# In[135]:


# Calculate false postive rate - predicting Converted when customer does not have Converted
print(FP/ float(TN+FP))


# In[136]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[137]:


# Negative predictive value
print(TN / float(TN+ FN))


# In[138]:


predict_test = model.predict(X_test)


# #Plotting the ROC Curve

# In[139]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[140]:


draw_roc(y_test,predict_test)


# #Precision and Recall

# In[141]:


from sklearn.metrics import precision_score, recall_score
precision_score(y_test,predict_test)


# In[142]:


recall_score(y_test,predict_test)


# In[143]:


# predict the target on the test dataset
predict_test = model.predict(X_test)
print('Target on test data\n\n',predict_test)


# In[144]:


confusion2 = metrics.confusion_matrix(y_test, predict_test )
print(confusion2)


# In[145]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[146]:


# Let's check the overall accuracy.
testaccuracy= accuracy_score(y_test,predict_test)
testaccuracy


# In[147]:


# Let's see the sensitivity of our lmodel
testsensitivity=TP / float(TP+FN)
testsensitivity


# In[148]:


# Let us calculate specificity
testspecificity= TN / float(TN+FP)
testspecificity


# #Observation

# In[149]:


# Let us compare the values obtained for Train & Test:
print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))
print("Train Data Sensitivity :{} %".format(round((trainsensitivity*100),2)))
print("Train Data Specificity :{} %".format(round((trainspecificity*100),2)))
print("Test Data Accuracy     :{} %".format(round((testaccuracy*100),2)))
print("Test Data Sensitivity  :{} %".format(round((testsensitivity*100),2)))
print("Test Data Specificity  :{} %".format(round((testspecificity*100),2)))


# In[150]:


estimator = model.estimators_[5]


# In[151]:


X_list = list(X.columns)


# In[152]:


pip install graphviz


# In[153]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = model.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = X_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png'); 


# In[154]:


# Get numerical feature importances
importances = list(model.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[155]:


# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, X_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances'); 


# In[157]:



scores = cross_val_score(model, X, y)
print(scores.mean())

classifier = model.fit(X,y)
predictions = classifier.predict_proba(X_test)
print(predictions)


# In[ ]:




