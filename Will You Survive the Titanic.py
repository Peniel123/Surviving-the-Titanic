#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. The goal here is to build a predictive model that answers the question: “what sorts of people were more likely to survive?” by using passenger data (ie name, age, gender, socio-economic class, etc).

# # Data Acquisition
# 
# The data used for this project is gotten from Kaggle. It can be accessed from the link https://www.dropbox.com/sh/uaolfsybe2le8ox/AACDlEOTiB3Yk5AMwzXF64qLa?dl=0
# 
# The data into two - testing and training. 

# In[1]:


# import pandas library
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#import the testing and training data and assign them to variables
#url_testing_data = "https://www.dropbox.com/s/9z8zlj6zmj99nb1/test.csv?dl=0"
testing_data = pd.read_csv("C://Users/ilech/Downloads/titanic/test.csv")

#url_training_data = "https://www.dropbox.com/s/rzzs4of9zhd30wy/train.csv?dl=0"
training_data = pd.read_csv("C://Users/ilech/Downloads/titanic/train.csv")
training_data.head()


# In[2]:


testing_data = pd.read_csv("C://Users/ilech/Downloads/titanic/test.csv")
testing_data.head()


# In[3]:


# The dataframe.shape() and .describe() can be used to get an overall understanding and description of the tables of interest.
# .shape() shows the number of rows and columns respectively
training_data.shape


# In[4]:


# .describe() gives an overall description of the table
training_data.describe(include = "all")


# In[5]:


print(training_data.dtypes)


# # Data Cleaning/Wrangling

# The .describe() tabel shows a total of 891 passengers. Two columns have missing values - Age and Cabin. Age has 714 values thus indicating 177 missing values while Cabin has 687 missing values. Embarked has 2 missing values. We can verify this easily with the .isnull() function.

# In[6]:


#check for columns with missing data
training_data.isnull().sum()


# As we can see, there are 177 Missing values in Age column 687 missing values in Cabin column. We have to deal with these missing values in order to build a good machine learning model.

# ### Data Cleaning: Dealing with Missing Values
# To deal with the Age column, the missing values with be replaced with random numbers ranging around the mean. The range is calcuted using the standard deviation.

# In[7]:


#calculate the mean and standard deviation and assign it to a variable
mean = training_data["Age"].mean()
std = training_data["Age"].std()

#generate an array of random numbers ranging from mean-std to mean+std and assign it to a variable
random_age = np.random.randint(mean-std, mean+std, size = 177)
age_slice = training_data["Age"].copy()

#replace the missing values with the random numbers generated
age_slice[np.isnan(age_slice)] = random_age
training_data["Age"] = age_slice

#confirm that there are no missing values in Age
training_data.isnull().sum()


# ### Data Cleaning: Dropping Unnecessary Columns

# Some columns simply cannot be analysed and should be dropped immediately for simplicity. Columns to be dropped are "PassengerId", "Ticket", "Cabin", "Name"
# 
# "PassengerId", "Ticket", "Name" are all unique vaariables and as such cannot be analysed. "Cabin" can be analysed when the deck is extracted from it but since only 1st class passengers have cabins, the rest are 'Unknown'. Given that 687 values from "Cabin" are missing, the whole column will be removed

# In[8]:


#assign the columns to be dropped to a variable
droped_columns = ["PassengerId", "Ticket", "Cabin", "Name"]
#pass the variable into the .drop function
training_data.drop(droped_columns, axis=1, inplace=True)

#verify that the columns have been dropped
training_data.head(10)


# In[9]:


training_data["Embarked"].value_counts()


# In[10]:


# Filling 2 missing values with most frequent value
training_data["Embarked"] = training_data["Embarked"].fillna('S')


# ### Data Cleaning: Convert Categorical Data to Dummy Variable
# 
# Categorical data needs to be converted to numeric values so they can be used for regression analysis in the later on. In this case, the "Sex" and "Embarked" are the categorical variables assigned dummy variables.

# In[11]:


# create a dummy variable for Sex and Embarked using .get_dummies function
sex_dummy = pd.get_dummies(training_data["Sex"])

embarked_dummy = pd.get_dummies(training_data["Embarked"])

#For clarity, change column names in Embarked
embarked_dummy.rename(columns={'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'}, inplace=True)
embarked_dummy.head()


# In[12]:


# merge dataframe "embarked_dummy" and "training_data" 
training_data = pd.concat([training_data, embarked_dummy, sex_dummy], axis=1)

#drop original column "Sex" and "Embarked" from "training_data"
#training_data.drop(columns=["Sex", "Embarked"], inplace=True)
#training_data.head()


# # Feature Engineering
# 
# New features can be created as a linear combinations of features. One thing to consider is the size of a person’s family which is the sum of their ‘SibSp’ and ‘Parch’ attributes. Did people traveling alone have a better chance of survival? Or perhaps travelling with a family gave a higher chance? We'll examine both.

# In[13]:


#Creating new family_size column
training_data['Family_Size'] = training_data['SibSp'] + training_data['Parch']
training_data.head(10)


# # Exploratory Data Analysis
# 
# We can begin by calculating the correlation between variables of type "int64" or "float64" using the method .corr()

# In[14]:


training_data.corr()


# Next, we visualise the correlation between each variable and the target variable - "Survived"

# ## 1. Age

# In[15]:


ageplot = sns.FacetGrid(training_data, col="Survived", height = 7)
ageplot = ageplot.map(sns.distplot, "Age")
ageplot = ageplot.set_ylabels("Survival Probability")


# Conclusions drawn from this graphs alone are inconclusive. The age range of people with a high chance of survival range between 18-40 after which the chances of survival decreases as age increases. However, the same can be said for those who did not survive. The age range of people with a high chance of not surviving range between 18-40 after which the chances of dying decreases as age increases. 
# 
# To get a better understanding, a combination of the age and sex of survival may reveal more insights.

# In[16]:


sns.displot(data=training_data, x="Age", hue="male", col="Survived", kind="hist", height = 7)


# From the graph, it is clear that while the chances of surviving and dying are fairly similar across age range, the key difference is the sex composition of each. Those who did not survive comprised mostly of men while those who survived comprised mostly of women. This is true across all ages except little children.

# ## 2. SibSp (Number of Sibilings), Parch (Parents or children) and Family Size

# First, we visualize SibSp variable against the target variable - Survived. Then we look at Parch variable and family size (which is a combination of both SibSp and Parch) to see if any significant insight pops out

# In[17]:


bargraph_sibsp = sns.catplot(x = "SibSp", y = "Survived", data = training_data, kind="bar", height = 8)


# We can see that individuals with 1 or 2 sibilings have a good chance of survival.

# In[18]:


bargraph_Parch = sns.catplot(x = "Parch", y = "Survived", data = training_data, kind="bar", height = 8)


# We can see that individuals with 1-3 family members had a significant advantage over those with 0,4 or 5 family members.

# In[19]:


bargraph_Family_Size = sns.catplot(x = "Family_Size", y = "Survived", data = training_data, kind="bar", height = 8)


# The graph depicting the survival rate for each family size is fairly consistent with the previous graphs. Indivduals with 1-3 family members have significant advantage over those with higher or less family members.

# ## 3. Sex
# 
# Having examined sex along with age, it is still important to examine the influence of sex and the chances of survival alone

# In[20]:


women = training_data.loc[training_data.female == 1]["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# In[21]:


men = training_data.loc[training_data.male == 1]["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


# While ~78% of women survived the titanice, only ~18% of men did. This shows the overwhelming influence that sex played in surviving the titanic

# ## 4. Pclass

# In[22]:


pclassplot = sns.catplot(x = "Pclass", y="Survived", data = training_data, kind="bar", height = 7)


# This shows that higher class had a higher chance of survival

# ## PClass v Sex

# In[23]:


a = sns.catplot(x = "Pclass", y="Survived", hue="Sex", data=training_data, height = 7, kind="bar")


# This shows that depsite class differences, women in lower classes had higher chances of surviving than men in higher classes. This shows just how important sex was in surviving the titanic

# ## 5. Embarked

# In[24]:


sns.catplot(x="Embarked", y="Survived", data=training_data, height = 7, kind="bar")


# ## Embarked v PClass

# In[25]:


sns.catplot(x="Pclass", col="Embarked", data = training_data, kind="count", height=7)


# Passengers embarked from C station, majority of them was from 1st class. That's why we got survival probability of C embarked passengers higher.

# # The Machine Learning Model

# In[26]:


# establish the independent variable
train_x = training_data[['Pclass', 'male', 'female', 'Age', 'SibSp', 'Parch', 'Family_Size', 'Cherbourg', 'Queenstown', 'Southampton']]

# establish the dependent variable
train_y = training_data[['Survived']]

#split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.20, random_state=42)


# ### 1. Random Forest Classifier 

# In[27]:


clf1 = RandomForestClassifier()
clf1.fit(x_train, y_train)
rfc_y_pred = clf1.predict(x_test)
rfc_accuracy = accuracy_score(y_test,rfc_y_pred) * 100
print("accuracy=",rfc_accuracy)


# ### 2. Logistic Regression

# In[28]:


clf2 = LogisticRegression()
clf2.fit(x_train, y_train)
lr_y_pred = clf2.predict(x_test)
lr_accuracy = accuracy_score(y_test,lr_y_pred)*100

print("accuracy=",lr_accuracy)


# ### 3. K-Neighbor Classifier

# In[29]:


clf3 = KNeighborsClassifier(5)
clf3.fit(x_train, y_train)
knc_y_pred = clf3.predict(x_test)
knc_accuracy = accuracy_score(y_test,knc_y_pred)*100

print("accuracy=",knc_accuracy)


# Since we're getting maximum accuracy score with Logistics Regression, we choose it for making predictions on test.csv.

# In[30]:


testing_data.isnull().sum()


# In[31]:


#calculate the mean and standard deviation and assign it to a variable
mean_test = testing_data["Age"].mean()
std_test = testing_data["Age"].std()

#generate an array of random numbers ranging from mean-std to mean+std and assign it to a variable
random_age_test = np.random.randint(mean_test-std, mean_test+std, size = 86)
age_slice_test = testing_data["Age"].copy()

#replace the missing values with the random numbers generated
age_slice_test[np.isnan(age_slice_test)] = random_age_test
testing_data["Age"] = age_slice_test

#confirm that there are no missing values in Age
testing_data.isnull().sum()


# In[32]:


# Replacing missing value of Fare column
testing_data['Fare'].fillna(testing_data['Fare'].mean(), inplace=True)

testing_data.isnull().sum()

# create a dummy variable for Sex and Embarked using .get_dummies function
sex_dummy = pd.get_dummies(testing_data["Sex"])

embarked_dummy = pd.get_dummies(testing_data["Embarked"])

#For clarity, change column names in Embarked
embarked_dummy.rename(columns={'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'}, inplace=True)
embarked_dummy.head()


# In[33]:


# merge dataframe "embarked_dummy" and "testing_data" 
testing_data = pd.concat([testing_data, embarked_dummy, sex_dummy], axis=1)


# In[34]:


col_to_drop = ["PassengerId", "Ticket", "Cabin", "Name", "Sex", "Embarked"]
testing_data.drop(col_to_drop, axis=1, inplace=True)
testing_data.head(10)


# In[37]:


x_test = testing_data
y_pred = clf1.predict(x_test)
originaltest_data = pd.read_csv("C://Users/ilech/Downloads/titanic/test.csv")
submission = pd.DataFrame({
        "PassengerId": originaltest_data["PassengerId"],
        "Survived": y_pred
    })
submission.head(20)

