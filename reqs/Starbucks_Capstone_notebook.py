#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge - Udacity Project
# 
# ### Introduction
# 
# The data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="picA.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="picB.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# ----

# # Problem Statement
# The goal of this project is to combine transaction, demographic, and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# This goal can be achieved by following the below-mentioned strategies:
# 
# * Exploring and Visualizaing the Data.
# * Applying Quick Data Analysis.
# * Preprocessing the data.
# * Scaling the numerical features.
# * Trying several Supervised Learning Models.
# * Evaluating the models using the chosen metric (Accuracy)- Choosing the best model among them.
# * If the results need to be improved, implementing GridSearchCV to find the best parameters (in order to improve the performance of the chosen model).
# 

# ----

# # Metrics
# 
# In order to evaluate our model's performance, we will use accuracy. This Metric was chosen for the following reasons :
# * Since we have a simple classification problem, i.e. either: 
#   * offer viewed
#   * offer completed.
# 
# * It enables us to recognize HOW WELL our model is predicting by comparing the number of correct predictions with the total number of predictions ( the concept of accuracy).

# ----

# # Exploratory Data Analysis

# ----

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import math
import json
import datetime
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

get_ipython().run_line_magic('matplotlib', 'inline')

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# In[2]:


sns.set()


# ----

# ### 1. Portfolio dataset

# In[3]:


portfolio.head()


# In[4]:


# checking the columns' titles and datatypes 
portfolio.info()


# In[5]:


# checking the number of rows and columns of the dataset
portfolio.shape


# In[6]:


# checking for the existence of missing values(NaNs)
portfolio.isna().sum()


# In[7]:


# checking the offer types the customer can receive
portfolio['offer_type'].unique()


# In[8]:


# checking for duplicates
portfolio.columns.duplicated().sum()


# In[9]:


# checking the number of unique offers
portfolio['id'].nunique()


# In[10]:


# grouping the offers by their type 
portfolio.groupby('offer_type')['id'].count()


# ### The above preliminary Exploration for the Portfolio Dataset shows the following:
# 1. The dataset has 6 columns and 10 rows.
# 2. This dataset has no null values nor duplicates.
# 3. There are three types of offers : 'bogo'(Buy One Get One free), 'informational' and 'discount'.
# 4. There are 4 offers included in the dataset that are classified as : “bogo“ , 4 offers classified as : ”discount” and 2 offers classified as : “informational”.
# 5. The 'difficulty' column unit is dollars , which does not reflect how difficult to be rewarded. Rescaling this feature is a useful step to do. This needs to be done before Modeling.

# ----

# ### 2. Profile dataset

# In[11]:


profile.head()


# In[12]:


# checking the number of rows and columns of the dataset
profile.shape


# In[13]:


# checking for the existence of missing values(NaNs)
profile.isna().sum()


# In[14]:


# checking the columns' titles and datatypes 
profile.info()


# In[15]:


# checking for duplicates
profile.columns.duplicated().sum()


# In[16]:


# checking the number unique customers
profile['id'].nunique()


# In[17]:


# checking the unique values of difficulty column
uniq_dif_sorted = np.sort(portfolio.difficulty.unique())
uniq_dif_sorted


# In[18]:


# checking the unique values of the 'gender' column
profile['gender'].unique()


# In[19]:


profile[profile['age']==118].count()


# In[20]:


plt.hist(profile['age'], bins=10);


# In[21]:


# checking the number of Male and Famale customers 
profile.gender.value_counts()


# In[22]:


profile_gender_counts = profile.gender.value_counts()
x = ['M','F','O']
data = profile_gender_counts
plt.bar(x,height = data);
xlocs, xlabs = plt.xticks()
for i, v in enumerate(data):
    plt.text(xlocs[i] - 0.13, v , str(v))
plt.xlabel('Gender Type');
plt.ylabel('Count');
plt.title('The Number of Customers in Each Gender Type');


# In[23]:


# compute the percentages of the gender distribution
total_counts = profile_gender_counts.sum()
gender_perc = round((profile_gender_counts/ total_counts)*100,2)
gender_perc


# In[24]:


# getting the statitical summary of the 'income' column
profile['income'].describe()


# In[25]:


# checking the distribution of 'income' column
profile['income'].hist();


# In[26]:


print('1- Number of customers with income = $76,000 is:', profile.age[profile['income']== 76000.0].count())
print('2- Number of customers with income = $75,000 is:', profile.age[profile['income']== 75000.0].count())
print('3- Number of customers with income = $50,000 is:', profile.age[profile['income']== 50000.0].count())
print('4- Number of customers with income = $49,000 is:', profile.age[profile['income']== 49000.0].count())


# In[27]:


# getting the statitical summary of the 'age' column
profile['age'].describe()


# In[28]:


# checking the unique values in the customers ages sorted in descending order
print(-np.sort(-profile['age'].unique()))


# In[29]:


# get the count of each unique value in the 'age' column and sort that count in an descending order
# this will allow us to identify whether the value is an outlier or not
profile['age'].value_counts().sort_values(ascending=False)


# In[30]:


# checking the distribution of 'age' column
profile['age'].hist();


# In[31]:


# checking the number of customers that are registered at the age = 118
profile['age'][profile['age'] == 118].count()


# In[32]:


# checking the count of values of the rows into which the customers age = 118
profile[profile['age']==118].count()


# It is clear that customers with age 118 has no values on both the 'gender' and 'income' columns. To double check this we would do the following:

# In[33]:


# creating a dataframe with only the customers with age = 118 
# this data frame will include the coressponding gender and income columns to the customers with age = 118
df_118 = profile[['gender','income','age']][profile['age']==118]


# In[34]:


# getting a quick look on the profile data of customers registered at age =118 
print(df_118.head())
print('1-The shape of this dataframe is' ,df_118.shape)
print('2-The number of null values in the "gender" column is:', df_118['gender'].isnull().sum())
print('3-The number of null values in the "income" column is:', df_118['income'].isnull().sum())


# ### The above Exploration and Visualization for the Profile Dataset shows the following:
# * The dataset has 5 columns and 17,000 rows.
# * The dataset has no duplicated rows.
# * The dataset has 2175 missing values on each of: ‘gender’, ’income’ variables.
# * The customers ages range from 18 to 101. Although that 2175 customers were registered at age 118 but I stilI considered this specific age an outlier b/c it appears clearly that there is something wrong related with these 2175 rows in the dataset.
# * Exploring and visualizing three variables in this dataset: ‘gender’,’income’ and ’age’, allowed me to get the following conclusion:
#   * The missing values in 'gender' and ‘income’ variables which are are related solely and specifically with the 2175 customers registered at age 118. In other words, customers at age 118 has no registered ‘gender’ and ‘income’. This needs to be cleaned in the Data Preprocessing (Wrangling/Cleaning) Section.
#   * Customers income ranges from 30,000 and 120,000 with most of the customers’ incomes fall between 50,000 and 75,0000.
#   * According to the available data, There are three ‘gender’ categories into which the customers falls in ( M, F and O). Keeping in our mind the above observation that there are 2175 missing values, Male Customers (8484 men) are more than Female Customers(6129 women) with 57% of customers are Males compared to 41% Females. However, there are 212 customers chose “O” as their gender.
# 

# ----

# ### 3. Transcript Dataset

# In[35]:


transcript.head()


# In[36]:


# checking the number of rows and columns of the dataset
transcript.shape


# In[37]:


# checking for duplicates
transcript.columns.duplicated().sum()


# In[38]:


# checking the columns' titles and datatypes 
transcript.info()


# In[39]:


# checking for the existence of missing values(NaNs)
transcript.isna().sum()


# In[40]:


# getting the types of events of the transcripts 
transcript.event.unique()


# In[41]:


# checking the count of each event type
transcript.event.value_counts()


# In[42]:


# creating a dataframe to include ONLY the 'transaction' event
df_transaction = transcript[transcript['event'] == 'transaction']


# In[43]:


#getting a random sample of 'value' column
df_transaction['value'].sample(100)


# ### The above Exploration for the Transcript Dataset shows the following:
# * The dataset has 4 columns and 306,534 rows.
# * The dataset has no duplicated rows nor missing values.
# * The ‘value’ column is a dictionary in which we can apply some kind of Feature Engineering to extract useful data that would surely contribute in the success of our future model. This step will be done through the Data Preprocessing (Wrangling/Cleaning) Section.
# * There are four types of events in this dataset: ‘transaction’, ’ offer received’, ‘offer viewed’ and ‘offer completed’.
# * All the events that are classified as ‘transaction’ do not have an ‘offerid’ within its ‘value’ column.

# ----

# # Data Processing

# ----

# ### Portfolio Dataset
# 
# Steps that needs to be done:
# * Rename 'id' column to 'offer_id'.
# * Change the unit of 'duration' column from days to hours.
# * Rename 'duration' column to 'duration_h' representing that the unit of measurment is 'hours'
# * Normalize 'difficulty' and 'reward' features using the MinMaxScaler.
# * Create dummy variables from the 'channels' column using one-hot encoding then Drop the 'channels' column.
# * Replace the 'offer_id' by more easy ids.
# * Replace the 'offer_type' by integers representing each offer type as follow:
#   1.  bogo
#   2.  discount
#   3.  informational

# In[44]:


# creating a copy from the dataset to be cleaned
clean_portfolio = portfolio.copy()


# In[45]:


clean_portfolio.head()


# #### Rename 'id' column to 'offer_id'.

# In[46]:


# renaming 'id' column to offer_id.
clean_portfolio.rename(columns={'id':'offer_id'},inplace=True)


# #### Change the unit of 'duration' column from days to hours.

# In[47]:


# changing the unit of 'duration' column from days to hours
clean_portfolio['duration'] = clean_portfolio['duration']*24


# #### Rename 'duration' column to 'duration_h' representing that the unit of measurment is 'hours'

# In[48]:


# renaming 'duration' column to 'duration_h' representing that the unit of measurment is 'hours'
clean_portfolio.rename(columns={'duration':'duration_h'},inplace=True)


# #### Normalize 'difficulty' and 'reward' features using the MinMaxScaler.

# In[49]:


# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['difficulty','reward']

#features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
clean_portfolio[numerical] = scaler.fit_transform(clean_portfolio[numerical])

# Show an example of a record with scaling applied
clean_portfolio.head()


# #### Create dummy variables from the 'channels' column using one-hot encoding then Drop the 'channels' column.

# In[50]:


# checking the channels options to decide on the number and the titles of the dummy variables to be created
clean_portfolio['channels'].head()


# In[51]:


# creating dummy variables from the 'channels' column 
clean_portfolio['channel_email'] = clean_portfolio['channels'].apply(lambda x: 1 if 'email' in x else 0)
clean_portfolio['channel_mobile'] = clean_portfolio['channels'].apply(lambda x: 1 if 'mobile' in x else 0)
clean_portfolio['channel_social'] = clean_portfolio['channels'].apply(lambda x: 1 if 'social' in x else 0)
clean_portfolio['channel_web'] = clean_portfolio['channels'].apply(lambda x: 1 if 'web' in x else 0)


# In[52]:


# checking that the dummy variables are correctly created 
clean_portfolio[['channels','channel_email','channel_mobile','channel_web','channel_social']].head()


# In[53]:


# dropping the 'channels' column
clean_portfolio.drop('channels', axis=1, inplace=True)


# In[54]:


clean_portfolio.head()


# #### Replace the 'offer_id' by more easy ids.

# In[55]:


# replacing the 'offer_id' by more easy ids
labels_offer_id = clean_portfolio['offer_id'].astype('category').cat.categories.tolist()
replace_map_comp_offer_id = {'offer_id' : {k: v for k,v in zip(labels_offer_id,list(range(1,len(labels_offer_id)+1)))}}


# In[56]:


# checking the new offer ids labels 
replace_map_comp_offer_id


# In[57]:


# replacing the categorical values in the 'offer_id' column by numberical values
clean_portfolio.replace(replace_map_comp_offer_id, inplace=True)


# #### Replace the 'offer_type' by integers representing each offer type

# In[58]:


# replacing the 'offer_type' by integers representing each offer type
labels_offer_type = clean_portfolio['offer_type'].astype('category').cat.categories.tolist()
replace_map_comp_offer_type = {'offer_type' : {k: v for k,v in zip(labels_offer_type,list(range(1,len(labels_offer_type)+1)))}}


# In[59]:


# checking the new offer types labels
print(replace_map_comp_offer_type)


# In[60]:


# replacing the categorical values in the 'offer_type' column by numberical values
clean_portfolio.replace(replace_map_comp_offer_type, inplace=True)


# In[61]:


# confirming changes
print(clean_portfolio.columns)
clean_portfolio.info()


# In[62]:


clean_portfolio.head()


# ----

# ### Profile Dataset

# Steps that need to be performed:
# * Preprocessing 'id' Feature:
# 
#   * Rename 'id' column name to 'customer_id'.
#   * Re-arrange the columns to have 'customer_id' column the first column in dataset.
#   * Replace the customer_id string values with easiest numerical values.
# 
# * Preprocessing 'age' Feature:
# 
#   * Replace age = 118 by NaN value.
#   * Remove customers (drop rows) with no 'age', 'gender' and 'income'. 
#   * Change the datatype of 'age' and 'income' columns to 'int'.
#   * Create a new column 'age_group' that includes the age_group to which each customer belongs
#   * Replace the 'age_group' categorical label by a corresponding numerical label, as follows:
#     * 1 : teenager
#     * 2 : young-adult
#     * 3 : adult
#     * 4 : elderly
# 
# * Preprocessing 'income' Feature:
# 
#   * Create a new column 'income_range' that includes the income-range to which the customer's income belongs.
#   * Replace the 'income_range' categorical label by a corresponding numerical label, as follows:
#     * 1 : average (30,000 - 60,000)
#     * 2 : above-average (60,0001 - 90,000)
#     * 3 : high (more than 90,000)
# 
# * Preprocessing 'gender' Feature:
# 
#   * Replace the 'gender' categorical labels with coressponding numerical label, as follows:
#     * 1 : F (Female)
#     * 2 : M (Male)
#     * 3 : O
# 
# * Preprocessing 'became_member_on' Feature:
# 
#   * Change the datatype of 'became_member_on' column from int to date and put it in the appropriate format in order to have a readable date format that can be analyzed easily if requiered.
#   * Add a new column 'start_year', that will present the year at which the customer become a member, to the existing dataset (for further analysis).
#   * Add a new column 'membership_days' ,that will present the number of days since the customer became a member, to the existing dataset (for further analysis).
#   * Create a new column 'member_type' representing the type of the member: new, regular or loyal depending on the number of his 'membership_days'.
#   * Replace the 'member_type' categorical label by a corresponding numerical label, as follows:
#     * 1 : new (member since 1000 days or less)
#     * 2 : regular (1001 - 1,600 days of membership)
#     * 3 : loyal (more than 1,600 days of membership)
#   * Drop 'age','income', 'became_member_on' and 'membership_days' columns, since they are no longer needed.

# In[63]:


# creating a copy from the dataset to be cleaned
clean_profile = profile.copy()


# #### Preprocessing 'id' Feature

# In[64]:


clean_profile.head()


# In[65]:


# renaming 'id' column name to 'customer_id'.
clean_profile.rename(columns={'id':'customer_id'},inplace=True)


# In[66]:


# checking the existing columns' order
clean_profile.columns


# In[67]:


# Re-arranging the columns to have 'customer_id' column the first column in dataset
clean_profile = clean_profile.reindex(columns=['customer_id', 'age', 'became_member_on', 'gender', 'income'])


# In[68]:


# confirming changes in columns order
clean_profile.columns


# In[69]:


# replacing the 'customer_id' string values  with easiest numerical values
labels_customer_id = clean_profile['customer_id'].astype('category').cat.categories.tolist()
replace_map_comp_customer_id = {'customer_id' : {k: v for k,v in zip(labels_customer_id,list(range(1,len(labels_customer_id)+1)))}}


# In[70]:


# replacing the  categorical labels in 'customer_id' column with numerical labels
clean_profile.replace(replace_map_comp_customer_id, inplace=True)


# In[71]:


clean_profile.head()


# #### Preprocessing 'age' Feature

# In[72]:


# replacing the age = 118 by NaN value
clean_profile['age'] = clean_profile['age'].apply(lambda x: np.nan if x == 118 else x)


# In[73]:


# checking that the age = 118 does not longer exist, the output should be nothing 
clean_profile[clean_profile['age'] == 118]


# In[74]:


# dropping rows with NaNs in 'age', 'gender' and 'income' columns
clean_profile.dropna(inplace=True)


# In[75]:


# checking that the rows with missing values(NaNs) have been successfully dropped
clean_profile.isna().sum()


# In[76]:


# changing the datatype of 'age' and 'income' columns to 'int'
clean_profile[['age','income']] = clean_profile[['age','income']].astype(int)


# In[77]:


# creating a new column representing the age group to which the customer belongs 
clean_profile['age_group'] = pd.cut(clean_profile['age'], bins=[17, 22, 35, 60, 103],labels=['teenager', 'young-adult', 'adult', 'elderly'])


# In[78]:


# checking the unique values in the newely created column
clean_profile['age_group'].head()


# In[79]:


# replacing the 'age_group' categorical labels by numerical labels
labels_age_group = clean_profile['age_group'].astype('category').cat.categories.tolist()
replace_map_comp_age_group = {'age_group' : {k: v for k,v in zip(labels_age_group,list(range(1,len(labels_age_group)+1)))}}


# In[80]:


print(replace_map_comp_age_group)


# In[81]:


# replace categorical labels in 'age_group' column with numerical labels
clean_profile.replace(replace_map_comp_age_group, inplace=True)


# In[82]:


# confirming that the replacement has been correctly performed 
clean_profile['age_group'].head()


# In[83]:


clean_profile.head()


# #### Preprocessing 'income' Feature

# In[84]:


# creating a new column representing the age group to which the customer belongs 
clean_profile['income_range'] = pd.cut(clean_profile['income'], bins=[29999, 60000, 90000, 120001],labels=['average', 'above-average', 'high'])


# In[85]:


# replacing the 'income_range' categorical labels by numerical labels
labels_income_range = clean_profile['income_range'].astype('category').cat.categories.tolist()
replace_map_comp_income_range = {'income_range' : {k: v for k,v in zip(labels_income_range,list(range(1,len(labels_income_range)+1)))}}


# In[86]:


# checking the categorical labels and its corresponding numerical labels for 'income_range' column
replace_map_comp_income_range


# In[87]:


# replacing categorical labels in 'income_range' column with numerical labels
clean_profile.replace(replace_map_comp_income_range, inplace=True)


# In[88]:


clean_profile.head()


# #### Preprocessing 'gender' Feature

# In[89]:


# replacing the 'gender' categorical labels with coressponding numerical label
labels_gender = clean_profile['gender'].astype('category').cat.categories.tolist()
replace_map_comp_gender = {'gender' : {k: v for k,v in zip(labels_gender,list(range(1,len(labels_gender)+1)))}}
clean_profile.replace(replace_map_comp_gender, inplace=True)


# In[90]:


# checking the numerical label and its corresponding categorical label
print(replace_map_comp_gender)


# In[91]:


clean_profile.head()


# #### Preprocessing 'membership_days' Feature

# In[92]:


# changing the datatype of 'became_member_on' column from int to date and put it in the appropriate format
clean_profile['became_member_on'] = pd.to_datetime(clean_profile['became_member_on'], format = '%Y%m%d')


# In[93]:


# adding a new column 'start_year', that will present the year at which the customer became a member
clean_profile['membership_year'] = clean_profile['became_member_on'].dt.year


# In[94]:


# adding a new column 'membership_days' ,that will present the number of days since the customer become a member
clean_profile['membership_days'] = datetime.datetime.today().date() - clean_profile['became_member_on'].dt.date

# removing the 'days' unit
clean_profile['membership_days'] = clean_profile['membership_days'].dt.days


# In[95]:


clean_profile.head()


# In[96]:


# creating a new column 'member_type' representing the type of the member: new, regular or loyal depending on the number of his 'membership_days'
clean_profile['member_type'] = pd.cut(clean_profile['membership_days'], bins=[390, 1000, 1600, 2500],labels=['new', 'regular', 'loyal'])


# In[97]:


# replacing the 'member_type' categorical labels by numerical labels
labels_member_type = clean_profile['member_type'].astype('category').cat.categories.tolist()
replace_map_comp_member_type = {'member_type' : {k: v for k,v in zip(labels_member_type,list(range(1,len(labels_member_type)+1)))}}


# In[98]:


# checking the numerical label and its corresponding categorical label
print(replace_map_comp_member_type)


# In[99]:


# replacing categorical labels in 'member_type' column with numerical labels
clean_profile.replace(replace_map_comp_member_type, inplace=True)


# In[100]:


# dropping 'age','income', 'became_member_on' and 'membership_days' columns, since they are no longer needed.
clean_profile.drop(columns = ['age','income','became_member_on', 'membership_days'], axis=1, inplace=True)


# In[101]:


# confirming changes
print(clean_profile.columns)
clean_profile.info()


# In[102]:


# check the first few rows of our preprocessed clean_profile dataset
clean_profile.head()


# ----

# ### Transcript Dataset

# Steps that need to be done:
# * Rename 'time' column to 'time_h' representing that the unit of measurment is 'hours'.
# * Preprocess 'person' Feature:
# 
#     * Rename 'person' column to 'customer_id'.
#     * Replace the categorical values in 'customer_id' column by the newly initiated numerical values corresponding with each customer id, which resulted from the previous preprocessing for 'id' feature
# 
# * Preprocess 'value' Feature:
# 
#     * Extract each key that exists in the 'value' column to a seperate column than dropping the 'value' column.
#     * Fill all the NaNs in the 'offer_id' column with 'N/A' values (i.e. Not Applicable).
#     * Drop the 'value' column since it is no longer needed.
# 
# * Preprocess 'event' Feature:
# 
#     * Excluding all events of 'transaction' or 'offer received' from our clean_transcript dataset.
#     * Replace the 'event' categorical labels with coressponding numerical label, as follows:
#       * 1 : offer completed
#       * 2 : offer viewed
# 
# * Preprocess 'offer_id' Feature:
# 
#     * Replace the categorical values in the 'offer_id' column by the corresponding numerical values used initiated during Preprocessing Portfolio Dataset

# In[103]:


# create a copy from the dataset to be cleaned
clean_transcript = transcript.copy()


# #### Rename 'time' column to 'time_h' representing that the unit of measurment is 'hours'.

# In[104]:


clean_transcript.head()


# In[105]:


# renaming 'time' column to 'time_h'
clean_transcript.rename(columns={'time':'time_h'},inplace=True)


# #### Preprocess 'person' Feature:

# In[106]:


# renaming 'person' column to 'customer_id'
clean_transcript.rename(columns={'person':'customer_id'},inplace=True)


# In[107]:


# replace categorical labels in 'customer_id' column with numerical labels
clean_transcript.replace(replace_map_comp_customer_id, inplace=True)


# In[108]:


# checking the first few entries in the 'customer_id' columns
clean_transcript['customer_id'].head()


# In[109]:


clean_transcript.head()


# The values that have not been replaced are for those customers who did not exist in the Profile Dataset. However, this issue will be automatically solved when we merge the Profile Dataset with the Transcript Dataset using the 'customer_id' column.
# 
# 

# #### Preprocessing 'value' Feature

# In[110]:


# Extract each key that exist in 'value' column to a seperate column.
# getting the different keys  that exists in the 'value' column
keys = []
for idx, row in clean_transcript.iterrows():
    for k in row['value']:
        if k in keys:
            continue
        else:
            keys.append(k)


# In[111]:


# checking the different keys of the 'value' dictionary
keys


# In[112]:


#create columns and specify the datatype of each of them
clean_transcript['offer_id'] = '' # datatype : string
clean_transcript['amount'] = 0  # datatype : integer
clean_transcript['reward'] = 0  # datatype : integer


# In[113]:


# Iterating over clean_transcript dataset and checking 'value' column
# then updating it and using the values to fill in the columns created above
for idx, row in clean_transcript.iterrows():
    for k in row['value']:
        if k == 'offer_id' or k == 'offer id': # b/c 'offer_id' and 'offer id' are representing the same thing 
            clean_transcript.at[idx, 'offer_id'] = row['value'][k]
        if k == 'amount':
            clean_transcript.at[idx, 'amount'] = row['value'][k]
        if k == 'reward':
            clean_transcript.at[idx, 'reward'] = row['value'][k]


# In[114]:


# filling all the NaNs in the 'offer_id' column with 'N/A' values (i.e. Not Applicable)
clean_transcript['offer_id'] = clean_transcript['offer_id'].apply(lambda x: 'N/A' if x == '' else x)


# In[115]:


# dropping the 'value' column 
clean_transcript.drop('value', axis=1, inplace=True)


# In[116]:


clean_transcript.head()


# #### Preprocessing 'event' Feature

# In[117]:


# checking the unique values in 'event' column
clean_transcript['event'].unique()


# #### Two important points to keep in mind:
# 
# * Since we are interested in the events related with the offers, i.e offer received, offer viewed, offer completed, we will remove all events of 'transaction' because they are not directly related with offers.
# * we will exclude all the events of 'offer recieved',since I want to focus on whether the customer:
#     * Only viewed the offer (offer viewed)
#     * viewed the offer and then completed it (offer viewed)&(offer completed).
# 
# 

# In[118]:


# excluding all events of 'transaction' from our clean_transcript dataset
clean_transcript = clean_transcript[clean_transcript['event'] != 'transaction']

# excluding all events of 'offer received' 
clean_transcript = clean_transcript[clean_transcript['event'] != 'offer received']


# In[119]:


# checking that the events of either 'transaction' or 'offer received'were successfully removed from the dataset
clean_transcript['event'].unique()


# In[120]:


# replacing the 'event' categorical labels with coressponding numerical label
labels_event = clean_transcript['event'].astype('category').cat.categories.tolist()
replace_map_comp_event = {'event' : {k: v for k,v in zip(labels_event,list(range(1,len(labels_event)+1)))}}


# In[121]:


# checking the numerical label and its corresponding categorical label
print(replace_map_comp_event)


# In[122]:


# replace categorical labels in 'event' column with numerical labels
clean_transcript.replace(replace_map_comp_event, inplace=True)


# In[123]:


# checking the current columns' datatypes 
clean_transcript.info()


# In[124]:


clean_transcript.head()


# #### Preprocessing 'offer_id' Feature

# In[125]:


# replacing the categorical values in the 'offer_id' column by its corresponding numerical values
clean_transcript.replace(replace_map_comp_offer_id, inplace=True)


# In[126]:


# confirming the changes done 
print(clean_transcript.columns)
clean_transcript.head()


# ----

# ### Merging the three clean datasets (Portfolio, Profile and Transaction ) into ONE Master Clean Dataset

# In[127]:


# merge 'clean_transcript' dataset with 'clean_portfolio' on 'offer_id'
master_df =clean_transcript.merge(clean_portfolio,how='left',on='offer_id')


# In[128]:


# join 'master_df' dataset with 'clean_profile' on 'customer_id'
master_df = master_df.merge(clean_profile,how ='left', on = 'customer_id')


# In[129]:


# checking our newely created master dataset
master_df.head()


# In[130]:


# check if we have any missing values 
master_df.info()


# In[131]:


# removing any row that contain NaNs
master_df = master_df.dropna(how='any',axis=0)


# In[132]:


# check if we have any missing values 
master_df.info()


# #### Quick Data Analysis on the Master DataSet
# A Quick data analysis would be performed on the master dataset to answer the following Questions :
# 
# 1- What is the most common offer each age group( teenagers, young-adults, adults and elderly)?
# 
# 2- Based on the demographic data of the customers who gets the highest income range , males or females?
# 
# 3- Who takes longer time to acheive each offer, Males or Females?
# 
# 4- How many new members Starbucks got each year?
# 
# 5- Which type of promotions(offers) each gender likes?
# 
# 6- What is the average length between two transcript for the same customer?
# 
# 7- From all the offers the customers viewed , how many offers they completed?
# 
# To easily understand the below analysis, Reconvert the values of the following features from numerical values to its original categorical values. The following maping for the features numerical values will be used:
# 
# * Mapping of Numerical values for ‘age_group’ feature:
#   * teenager
#   * young-adult
#   * adult
#   * elderly
# * Mapping of Numerical values for ‘income_range’ feature:
#   * average (30,000 - 60,000)
#   * above-average (60,0001 - 90,000)
#   * high (more than 90,000)
# * Mapping of Numerical values for ‘gender’ feature:
#   * F (Female) 
#   * M (Male)
#   * O
# * Mapping of Numerical values for 'offer_type' feature:
#   * bogo
#   * discount
#   * informational
# * Mapping of Numerical values for ‘event’ feature:
#   * offer completed
#   * offer viewed
# 

# In[133]:


# reconverting the values of the following features from numerical values to its original categorical values.
master_df['event'] = master_df['event'].map({1: 'Completed', 2: 'Viewed'})
master_df['offer_type'] = master_df['offer_type'].map({1: 'BOGO', 2: 'Discount', 3: 'Informational'})
master_df['income_range'] = master_df['income_range'].map({1: 'Average', 2: 'Above-Average', 3:'High'})
master_df['age_group'] = master_df['age_group'].map({1: 'teenager', 2: 'young-adult', 3:'adult', 4:'elderly'})


# ----

# ### Question 1 - What is the common offer each age group ( teenagers, young-adults, adults and elderly)?
# 
# 

# In[134]:


plt.figure(figsize=(14, 6))
g = sns.countplot(x="age_group", hue="offer_type", data=master_df)
plt.title('Most Popular Offers to Each Age Group')
plt.ylabel('Total')
plt.xlabel('Age Group')
xlabels = ['teenager','young-adult','adult','elderly']
g.set_xticklabels(xlabels)
plt.xticks(rotation = 0)
plt.legend(title='Offer Type')
plt.show();


# The most common offer type among all age groups is the BOGO , followed by the Discount Offers. Whereas, the least common offer to be sent is the informational offers. I believe that BOGO offers are more attractive compared to other offers provided by Starbucks.
# 
# 

# ----

# ### Question 2 - Based on the demographic data of the customers who gets the highest income range , males or females?

# In[135]:


plt.figure(figsize=(14, 6))
g = sns.countplot(x="gender", hue="income_range", data= master_df[master_df["gender"] != 3])
plt.title('Income Range vs Gender')
plt.ylabel('Income Range')
xlabels = ['Female', 'Male']
g.set_xticklabels(xlabels)
plt.xlabel('Gender')
plt.xticks(rotation = 0)
plt.show();


# Customers with High income (Above 90,000) are mostly female customers. Whereas, Average Income(30,000 - 60,000) customers are mostly males.

# ----

# ### Question 3 - Who takes longer time to acheive each offer, Males or Females?

# Males and Females are pretty close when it comes to the time spent to complete an offer. Both males and females take about 17 days to da so.

# In[136]:


tran_avg_len_g_f = master_df.groupby(['gender', 'offer_id'])['time_h'].mean().reset_index()
tran_avg_len_g_m = master_df.groupby(['gender', 'offer_id'])['time_h'].mean().reset_index()

print(tran_avg_len_g_f[tran_avg_len_g_f['gender'] == 1.]['time_h'].values[0], tran_avg_len_g_f[tran_avg_len_g_f['gender'] == 1.]['time_h'].values[0] / 24)
print(tran_avg_len_g_m[tran_avg_len_g_m['gender'] == 2.]['time_h'].values[0], tran_avg_len_g_m[tran_avg_len_g_m['gender'] == 2.]['time_h'].values[0] / 24)


# ----

# ### Question 4 - How many new members Starbucks got each year?
# 
# 

# In[137]:


master_df['membership_year'] = master_df['membership_year'].astype(int)


# In[169]:


plt.figure(figsize=(16, 6))
sns.countplot(x='membership_year', data = master_df)
plt.title('Number of Profiles In Each Year')
plt.ylabel('Number of Profiles')
plt.xlabel('Year')
plt.xticks()
plt.show();


# In[139]:


# getting the number of customers that became members on 2017
members_2017 = (master_df['membership_year']==2017).sum()

# getting the total number of members among all the available years
total = master_df['membership_year'].count()

# getting the percentages of customers that became members on 2017
perc_2017 = round((members_2017/total)*100,2)

print(members_2017)
perc_2017


# 2017 was the best year for Starbucks in terms of the number of new members. Around 38% of all the customers on our dataset regiseterd as members on this specific year.
# 
# 

# ----

# ### Question 5 - Which type of promotions(offers) each gender likes?

# In[140]:


plt.figure(figsize=(14, 6))
g = sns.countplot(x='gender', hue="offer_type", data= master_df[master_df["gender"] != 3])
plt.title('Most Popular Offers to Each Gender')
plt.ylabel('Total')
plt.xlabel('Gender')
xlabels = ['Female', 'Male']
g.set_xticklabels(xlabels)
plt.legend(title='Offer Type')
plt.show();


# The chart we got showed that both genders like BOGO and Discount offers and they have the same reaction toward Informational offers, they both seem to be not intersted to it.

# ----

# ### Question 6 - What is the average length between two transcript for the same customer?

# In[141]:


tran_avg_len = master_df.groupby(['customer_id', 'offer_id'])['time_h'].mean().reset_index()
tran_avg_len['time_h'].mean(), tran_avg_len['time_h'].mean() / 24


# The mean time it takes a customer to complete an offer is less than 16 days (372 hours).

# ### Question 7  - From all the offers the customers viewed , how many offers they completed?

# In[142]:


plt.figure(figsize=(14, 6))
g = sns.countplot(x='gender', hue="event", data= master_df[master_df["gender"] != 3])
plt.title('Most Popular Offers to Each Gender')
plt.ylabel('Total')
plt.xlabel('Gender')
xlabels = ['Female', 'Male']
g.set_xticklabels(xlabels)
plt.legend(title='Offer Type')
plt.show();


# ----

# In[143]:


total_trans_g_o = master_df[master_df["gender"] != 3].groupby(['gender','offer_type']).count()
total_trans_g_e = master_df[master_df["gender"] != 3].groupby(['gender','event']).count()
total_trans_go_o_t = total_trans_g_o.loc[(1)]['event'].sum()
total_trans_go_o_tt = total_trans_g_o.loc[(2)]['event'].sum()
total_trans_go_o_t_offers_f = total_trans_g_o.loc[(1)].loc[['BOGO', 'Discount', 'Informational']]['event'].sum()
total_trans_go_o_t_offers_m = total_trans_g_o.loc[(2)].loc[['BOGO', 'Discount', 'Informational']]['event'].sum()


# In[144]:


print('For Females:')
print(f"Number of offer completed: {total_trans_g_e.loc[(1, 'Completed')].values[0]}, {round((total_trans_g_e.loc[(1, 'Completed')].values[0]/total_trans_g_e.loc[(1, 'Viewed')].values[0])*100,2)}% of total offers viewed.")
print(f"Number of offer viewed: {total_trans_g_e.loc[(1, 'Viewed')].values[0]}.")
print("\n")
print('\nFor Males:')
print(f"Number of offer completed: {total_trans_g_e.loc[(2, 'Completed')].values[0]}, {round((total_trans_g_e.loc[(2, 'Completed')].values[0]/total_trans_g_e.loc[(2, 'Viewed')].values[0])*100,2)}% of total offers viewed.")
print(f"Number of offer viewed: {total_trans_g_e.loc[(2, 'Viewed')].values[0]}.")


# Females completed around 75% of the offers they viewed, it is 16% more than males who just completed 58% of the offers they viewed. Feamles seems to be convinced by the promotion easier than males.

# In[145]:


# Replacing the categorical values of the features by its corresponding numerical values, as before
labels_event1 = master_df['event'].astype('category').cat.categories.tolist()
replace_map_comp_event1 = {'event' : {k: v for k,v in zip(labels_event1,list(range(1,len(labels_event1)+1)))}}

labels_income1 = master_df['income_range'].astype('category').cat.categories.tolist()
replace_map_comp_income_range1 = {'income_range' : {k: v for k,v in zip(labels_income1,list(range(1,len(labels_income1)+1)))}}

labels_offer_type1 = master_df['offer_type'].astype('category').cat.categories.tolist()
replace_map_comp_offer_type1 = {'offer_type' : {k: v for k,v in zip(labels_offer_type1,list(range(1,len(labels_offer_type1)+1)))}}

master_df.replace(replace_map_comp_event1, inplace=True)
master_df.replace(replace_map_comp_offer_type1, inplace=True)
master_df.replace(replace_map_comp_income_range1, inplace=True)
master_df.replace(replace_map_comp_age_group, inplace=True)


# In[146]:


# confirming changes
master_df.head()


# ----

# # Build Models

# ----

# ### We build models to help us predict HOW will a given customer respond to an offer?

# In[147]:


master_df.columns


# But first, We need to split data into features and target labels, considering ONLY those features that we believe are important for our model to predict accurately.

# Those features are as follows:
# - time_h
# - offer_id
# - amount
# - reward_x ( Will be renamed to 'reward')
# - difficulty
# - duration_h
# - offer_type
# - gender
# - age_group
# - income_range
# - member_type
# 
# Our target is:
# 
# * 'event' that will be either:
#   * 1 : offer completed
#   * 2 : offer viewed

# In[148]:


# Rename 'reward_x' column to 'reward'
master_df.rename(columns ={'reward_x':'reward'}, inplace = True)


# In[149]:


# Split the data into features and target label
X = master_df[['time_h','offer_id','amount','reward','difficulty','duration_h','offer_type','gender','age_group','income_range', 'member_type']]
Y = master_df['event']


# In[150]:


X.head()


# In[151]:


Y.head()


# In[152]:


# normalizing some numerical values 
scaler = MinMaxScaler()
features = ['time_h', 'amount', 'reward', 'duration_h']
X_scaled = X.copy()
X_scaled[features] = scaler.fit_transform(X_scaled[features])
X_scaled.head()


# In[153]:


# creating training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)


# As mentioned in the Metric Section above, in order to evaluate our models performance , we will use accuracy. This Metric was chosen for the following reasons :
# 
# * since we have a simple classiifcation problem, i.e. either : offer viewed or offer completed.
# * It enables us to recognize HOW WELL our model is predicting by comparing the number of correct predictions witht the total number of predictions ( the concept of accuracy).

# In[154]:


# defining a function to calculate the accuracy for the models we will try below 
def predict_score(model):
    pred = model.predict(X_test)
    
    # Calculate the absolute errors
    errors = abs(pred - y_test)
    
    # Calculate mean absolute percentage error
    mean_APE = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mean_APE)
    
    return round(accuracy, 4)


# ----

# For Now, several models will be tried , then the best model along them would be chosen. Followed by an implementation for a GridSearch to find the best parameters ( in order to improve the performance of the chosen model)

# ### 1. Decision Tree

# In[155]:


dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
print(f'Accuracy of Decision Tree classifier on training set: {round(dt.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy: {predict_score(dt)}%')


# ### 2. Support Vector Machine

# In[156]:


svm = SVC(gamma = 'auto')

svm.fit(X_train, y_train)
print(f'Accuracy of SVM classifier on training set: {round(svm.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy: {predict_score(svm)}%')


# ### 3. Naive Bayes

# In[157]:


gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
print(f'Accuracy of SVM classifier on training set: {round(gnb.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy: {predict_score(gnb)}%')


# ### 4. Random Forest

# In[158]:


rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(X_train, y_train)
print(f'Accuracy of SVM classifier on training set: {round(rf.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy: {predict_score(rf)}%')


# ### 5. K-Nearest Neighbors

# In[159]:


knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
print(f'Accuracy of K-NN classifier on training set: {round(knn.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy: {predict_score(knn)}%')


# ### 6. LogisticRegression

# In[160]:


logreg = LogisticRegression()

logreg.fit(X_train, y_train)
print(f'Accuracy of Logistic regression classifier on training set: {round(logreg.score(X_train, y_train)*100,2)}%.')
print(f'Prediction Accuracy: {predict_score(logreg)}%')


# ----

# ## Model Evaluation

# In[161]:


# creating the variables that will be used to fill the results table
models = [svm, dt, gnb, knn, rf, logreg]
model_names = [type(n).__name__ for n in models]
training_accuracy = [x.score(X_train, y_train)*100 for x in models]
predection_accuracy = [predict_score(y) for y in models]


# In[162]:


# structuring a table to view the results of the different model tried above
results = [training_accuracy, predection_accuracy]
results_df = pd.DataFrame(results, columns = model_names, index=['Training Accuracy', 'Predicting Accuracy'])


# In[163]:


# show the results dataframe 
results_df


# The above table, shows the accuracy score related with using different models of supervised learning. 
# 
# As we see from the constructed table, we have a 100% accuracy for training as well as test for all thefour models we tested out of the six models. We avoid overfitting by choosing the lowest accuracy score on the test data for KNeighborsClassifier models. We chose this model because as a first look it can be used on a to solve a Binary outcome problem statement. On whther the customer will view the offer (just checking) or finish it (complete the offer).

# ## Model Refinment
# 

# We have not done it here, but based on the results we got any other moodel refinement may not help. We need to perform cross validation as well as Grid Search CV to improve the model/models. 

# # Conclusion

# #### For this project , we tried to analyze the datasets provided by Starbucks for the udacity project and then build a model that can predict whether a customer would complete the offer or just view it?
# 
# As the specific steps we explored each dataset, did exploratory data analysis, visualize it to get an overall understanding on the data. This also included analyzing different aspects of the datasets. Subsequently we performed the Preprocessing, Cleaning and Feature Engineering. Then we 
# created latent features that will have the ability to improve the performance of the model. These featured were actually derived from orginal existing column but with less range of values and simple values to include within the data set. Examples on that are the following:
# 
# - 'age_group' feature derived from 'age' feature. We then replaced the feature's categorical labels by a corresponding numerical label, as follows: 
#     - 1 : teenager
#     - 2 : young-adult
#     - 3 : adult
#     - 4 : elderly
# 
# -  'income_range' feature derived from 'income' feature. that includes the income-range to which the customer's income belongs. We replaced the 'income_range' categorical labels by corresponding numerical labels, as follows: 
#     - 1 : average (30,000 - 60,000)
#     - 2 : above-average (60,0001 - 90,000)
#     - 3 : high (more than 90,000)
# 
# - 'member_type' feature derived from 'became_member_on' feature. The features categorical labels has been replaced by corresponding numerical labels, as follows: 
#     - 1 : new (memebr since 1000 days or less)
#     - 2 : regular (1001 - 1,600 days of membership)
#     - 3 : loyal (more than 1,600 days of membership)
# 
# #### The analysis provided the following:
# 
# * Customers income ranges from 30,000 and 120,000 with most of the customers’ incomes fall between 50,000 and 75,0000.
# * According to the available data, there were three ‘gender’ categories into which the customers falls in ( M, F and O). Keeping in our mind the above observation that there are 2175 missing values, Male Customers (8484 men) are more than Female Customers(6129 women) with 57% of customers are Males compared to 41% Females. However, there are 212 customers chose “O” as their gender.
# * The most common offer type among all age groups is the BOGO , followed by the Discount Offers. Whereas, the least common offer to be sent is the informational offers. It can be surmised that BOGO offers are more attractive compared to other offers provided by Starbucks.
# * Customers with High income (Above 90,000) are mostly female customers. Whereas, Average Income(30,000 - 60,000) customers are mostly males.
# * Males and Females are pretty close when it comes to the time spent to complete an offer. Both males and females take about 17 days to da so.
# * 2017 was the best year for Starbucks in terms of the number of new members. Around %38 of all the customers on our dataset regiseterd as members on this specific year.
# * Both genders like BOGO and Discount offers and they have the same reaction toward Informational offers, they both seem to be not interested to it.
# * The mean time it takes a customer to complete an offer is less than 16 days (372 hours).
# * Females completed around 75% of the offers they viewed, it is 16% more than males who just completed 58% of the offers they viewed. Feamles seems to be convinced by the promotion easier than males.
# 

# # Next Steps

# We can get more insights from this data set and robust prediction models may be built to solve problems associated with this data set. Examples for these model are the following :
# 
# * Building a model that can predict which kind of offers to be sent to whcihc customer?
# * Building a model that can predict which customers would buy any way ( regardless if there is an offer or not)

# In[164]:


####!!jupyter nbconvert *.ipynb
get_ipython().getoutput('jupyter nbconvert --to html *.ipynb')


# In[ ]:




