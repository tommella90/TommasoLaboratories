#%%
### CASE STUDY LAB TOMMASO
import pandas as pd
import numpy as np


#%% Open the data and
from matplotlib import pyplot as plt

path = "C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/data/labs"
file1 = pd.read_csv(f'{path}/file1.csv')

list1 = []
for i in range(1,4):
    list1.append(pd.read_csv("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/data/labs/file" + str(i) + ".csv"))

file1 = pd.read_csv(f'{path}/file1.csv')
file2 = pd.read_csv(f'{path}/file2.csv')
file3 = pd.read_csv(f'{path}/file3.csv')

dfList = [file1, file2, file3]

def lower_case_column_names(df):
    df.columns=[i.lower() for i in df.columns]

list(map(lower_case_column_names, dfList))


#%% Aggregate data into one Data Frame using Pandas.
# Pay attention that files may have different names for the same column. therefore,
# make sure that you unify the columns names before concating them.
file1.rename(columns = {"st": "state"}, inplace=True)
file2.rename(columns = {"st": "state"}, inplace=True)

list1 = file1.columns
list2 = file2.columns
list3 = file3.columns

sorted(list1)
sorted(list2)
sorted(list3)


#%% Deleting and rearranging columns –
# delete the column customer as it is only a unique identifier for each row of data
df = pd.concat([file1,file2,file3], axis=0)
df.reset_index(drop=True, inplace=True)
df = df.drop(labels=['customer'], axis=1)


#%% Working with data types – Check the data types of all the columns and fix the incorrect ones
# (for ex. customer lifetime value and number of open complaints )
dropPerc = lambda x : x.replace ("%", "") if type(x) == str else x
df["customer lifetime value"] = df["customer lifetime value"].apply(dropPerc)
print(df['customer lifetime value'].isnull().sum())

'''
df['customer lifetime value'] = df['customer lifetime value'].astype(str)
df['customer lifetime value2'] = df['customer lifetime value'].str.strip('%')
df['customer lifetime value2']
print(df['customer lifetime value2'].isnull().sum())

df2 = df[['customer lifetime value', 'customer lifetime value2']]

df['customer lifetime value'] = df['customer lifetime value'].astype(str)
percBool = df['customer lifetime value'].str.contains("%", case=False)
df['customer lifetime value'] = df.loc[percBool, "customer lifetime value"].str.replace('%', '')
print(df['customer lifetime value'].isnull().sum())
'''


#%%
# clean the number of open complaints and extract the middle number which is changing between records.
# pay attention that the number of open complaints is a categorical feature

df["number of open complaints"] = df["number of open complaints"].astype(str)
myF = lambda x: x[2] if x.count('') > 4 else x
df["number of open complaints2"] = df["number of open complaints"].apply(myF)
fillNa = lambda x: np.nan if x=="nan" else x
df["number of open complaints2"] = df["number of open complaints2"].apply(fillNa)
print(df['number of open complaints2'].isnull().sum())

df["number of open complaints2"].unique()


#%%
# Filtering data and Correcting typos –
# Filter the data in state and gender column to standardize the texts in those columns
print(df.gender.unique())
df['gender'].replace('Femal', 'F', inplace=True)
df['gender'].replace('female', 'F', inplace=True)
df['gender'].replace('Male', 'M', inplace=True)
print(df.gender.unique())

df.state = list(map(lambda x: "Arizona" if x == "AZ" else x, df.state))
df.state = list(map(lambda x: "Washington" if x == "WA" else x, df.state))
print(df.state.unique())


#%%
df["customer lifetime value"] = df["customer lifetime value"].fillna(0)
df["customer lifetime value"] = df["customer lifetime value"].astype(float).astype(np.int64, errors='ignore')

df["customer lifetime value"] = list(map(lambda x: np.nan if x==0 else x, df["customer lifetime value"]))
for i in range(10):
    print(type(df["customer lifetime value"][i]))


#%% Removing duplicates
print(len(df.index))
df.drop_duplicates(subset=None, keep='first', inplace=True)
print(len(df.index))
df.reset_index(inplace=True)

#%%
# For the numerical variables, check the multicollinearity between the features.
# Please note that we will use the column total_claim_amount later as the target variable.
fig = plt.figure(figsize=(8,6))
sns.heatmap(df_num.corr(), cmap='vlag', annot=True)


#%% Replacing null values – Replace missing values with means of the column (for numerical columns).
# Pay attention that the Income feature for instance has 0s which is equivalent to null values.
# (We assume here that there is no such income with 0 as it refers to missing values)
# Hint: numpy.nan is considered of float64 data type.
from pandas.api.types import is_numeric_dtype
is_numeric_dtype(df['income'])

df['income'].replace(0, np.nan, inplace=True)
df['income'].fillna(df['income'].mean(), inplace=True)
df['income'] = df['income'].round().astype(int)

for i in df.columns:
    if is_numeric_dtype(df[i]) == True:
        df[i].fillna(df[i].mean(), inplace=True)


#%% Bucketing the data -
# Write a function to replace column "State" to different zones.
# California as West Region, Oregon as North West, and Washington as East,
# and Arizona and Nevada as Cent

df['region'] = ['W' if x=='California' \
                else 'NW' if x=='Oregon' \
                else 'E' if x=='Washington'\
                else 'C' if (x=='Arizona' or x=='Nevada')\
                else 'Nan' for x in df['state']]


#%%
def regionAssignment(state):
    state.replace("California", "WE", inplace=True)
    state.replace("Oregon", "NE", inplace=True)
    state.replace("Washington", "E", inplace=True)
    state.replace("Arizona", "C", inplace=True)
    state.replace("Nevada", "C", inplace=True)
    state.replace("Cali", "NaN", inplace=True)


#%% (Optional) Standardizing the data – Use string functions to standardize the text data (lower case)n
for i in df.columns:
    if is_numeric_dtype(df[i]) == False:
        df[i] = df[i].str.lower()







#%% Get the numeric data into dataframe called numerical and categorical columns in a dataframe
# called categoricals. (You can use np.number and np.object to select the numerical data types
# and categorical data types respectively)
path = "C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/data/labs"
df = pd.read_csv(f'{path}/Data_Marketing_Customer_Analysis_Round3.csv')
df = df.iloc[:,1:]
df_num = df.iloc[:,1:].select_dtypes(np.number)
df_cat = df.select_dtypes(object)


#%% Now we will try to check the normality of the numerical variables visually
# Use seaborn library to construct distribution plots for the numerical variables
# Use Matplotlib to construct histograms
# Do the distributions for different numerical variables look like a normal distribution?
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#
### seaborn distrib plots
listColor = ['green', 'red', 'yellow', 'blue', 'green', 'red', 'yellow', 'blue']

fig, ax = plt.subplots(4,2, figsize = (8,12))

for i in range(1,9):
    plt.subplot(4,2,i)
    sns.distplot(df_num.iloc[:,i-1], color = listColor[i-1])
    plt.title(f'{df_num.columns[i-1].upper()}')
fig.subplots_adjust(hspace=1.5)

#%%
### matplotlib histograms
fig, ax = plt.subplots(4,2, figsize = (10,15))
for i in range(1,9):
    plt.subplot(4,2,i)
    plt.hist(df_num.iloc[:,i-1], color = '#398f29', edgecolor = 'white', bins = 50)
    plt.title(f'{df_num.columns[i-1]}')


#%% Check correlation among all numerical variables
sns.pairplot(df_num, palette=None, kind='scatter',diag_kind='auto',
             height=2.5, aspect=1, size=None)


#%% Check correlation between Total claim amount and Monthly premium auto
fig = plt.figure(figsize=(8,6))
plt.scatter(df_num['total_claim_amount'], df_num['monthly_premium_auto'], s=.5)
plt.title("Correlation")
plt.xlabel("Total claim amount")
plt.ylabel("Monthly premium auto")

df3 = df[['total_claim_amount', 'monthly_premium_auto']]

#%%
fig, ax = plt.subplots(figsize = (6,6))
responses = df_cat['response'].value_counts()
ax.set_title('Total number of responses', fontweight='bold')
ax.bar(responses.index, responses, color=('khaki','lightblue'))


#%% Show a plot of the response by the sales hannel.
## We created categories of continuous variables by splitting them in quartiles
# This is a function to create a new columns representing the quartile range of the previous one
def createQuartiles(col):
    quartiles = col.describe()
    col = ['Q1' if x<quartiles['25%'] else 'Q2' if (x>=quartiles['25%'] and x<quartiles['50%'])
    else 'Q3' if (x>=quartiles['50%'] and x<=quartiles['75%']) else 'Q4' for x in col]
    return col


#%%
## apply the fuction to income and claim amount to divide them in quartiles groups
df['claimQuart'] = createQuartiles(df['total_claim_amount'])
df['incomeQuart'] = createQuartiles(df['income'])


#%% Optional: Drop one of the two features that show a high correlation between them (greater than 0.9)
# If there is no pair of features that have a high correlation, then do not drop any features.
# ANSWER: no correlation higher than .9
fig = plt.figure(figsize=(8,6))
plt.scatter(df_num['customer_lifetime_value'], df_num['monthly_premium_auto'], s=.5)
plt.title("Correlation")
plt.xlabel("customer_lifetime_value")
plt.ylabel("Monthly premium auto")


## It shows 9 lines, so I checked by number of policies
fig = plt.figure(figsize=(8,6))
sns.lmplot( x="customer_lifetime_value", y="monthly_premium_auto", data=df, fit_reg=False,
            hue='number_of_policies', markers='.', legend=False)
plt.legend(loc='lower right')
plt.title("Correlation")
plt.xlabel("customer_lifetime_value")
plt.ylabel("Monthly premium auto")

#%%
quartiles = df['sales_channel'].describe()
quartiles
col = ['Q1' if x<quartiles['25%'] \
           else 'Q2' if (x>=quartiles['25%'] and x<quartiles['50%']) \
    else 'Q3' if (x>=quartiles['50%'] and x<=quartiles['75%']) \
    else 'Q4' for x in col]
return col

#%%
explanVars = df[['total_claim_amount', 'region', 'education', 'employment_status', 'gender', \
                 'location_code', 'policy_type', 'vehicle_class', 'coverage', 'number_of_policies', 'number_of_open_complaints']]

fig5, ax5 = plt.subplots(6,2, figsize = (10,18))
for i in range(1,11):
    x1 = explanVars.groupby(explanVars.columns[i]).mean(['total_claim_amount'])
    plt.subplot(5,2,i)
    sns.barplot(x=x1.index, y='total_claim_amount', data=x1)
    plt.title(f'{explanVars.columns[i]}')
    fig.subplots_adjust(hspace=3)


#%%
#%%
fig2, ax2 = plt.subplots(1,3, figsize=(12,4))
fig.suptitle("RESPONSE PER GROUPS", fontsize=16)

sns.countplot(x='sales_channel', hue='response', data=df_cat, order=None, hue_order=None,
              orient=None, color=None, dodge=True, ax=ax2[0])
ax2[0].set_title('Sales Channel')

sns.countplot(x='claimQuart', hue='response', data=df, order=None, hue_order=None,
              orient=None, color=None, dodge=True, ax=ax2[1])
ax2[1].set_title('Number of claims (quartiles)')

sns.countplot(x='incomeQuart', hue='response', data=df, order=None, hue_order=None,
              orient=None, color=None, dodge=True, ax=ax2[2])
ax2[2].set_title('Income quartiles')

fig.subplots_adjust(hspace=4.5)

#%%
### plot them with bar charts
x1 = df.groupby(df['number_of_policies']).mean(['customer_lifetime_value'])
#sns.barplot(x=x1.index, y='number_of_policies', data=x1)
x1.reset_index(inplace=True)
x1['number_of_policies'].astype(str)
x1

fig, ax = plt.subplots(1,2,figsize = (12,4))

ax[0].set_title('Montly premium', fontsize=15)
ax[0].bar(x1['number_of_policies'], x1['monthly_premium_auto'])

ax[1].set_title('Customer lifetime value', fontsize=15)
ax[1].bar(x1['number_of_policies'], x1['customer_lifetime_value'])


#%% plot them singularly
col1 = explanVars.groupby(explanVars['employment_status']).mean(['total_claim_amount'])
col1 = col1.loc[['employed', 'unemployed']]
col2 = explanVars.groupby(explanVars['location_code']).mean(['total_claim_amount'])
col3 = explanVars.groupby(explanVars['coverage']).mean(['total_claim_amount'])
col4 = explanVars.groupby(explanVars['vehicle_class']).mean(['total_claim_amount'])
col2

fig, ax = plt.subplots(2,2,figsize = (12,8))
fig.suptitle("Potential explan. variables", fontsize=16)

sns.barplot(x=col1.index, y='total_claim_amount', data=col1, ax=ax[0,0])
ax[0,0].set_title('Employment stat.')
sns.barplot(x=col2.index, y='total_claim_amount', data=col2, ax=ax[0,1])
ax[0,1].set_title('Location')
sns.barplot(x=col3.index, y='total_claim_amount', data=col3, ax=ax[1,0])
ax[1,0].set_title('Coverage')
sns.barplot(x=col4.index, y='total_claim_amount', data=col4, ax=ax[1,1])
ax[1,1].set_title('vehicle_class')
fig.subplots_adjust(hspace=.5)


#%%
response_claim = df_cat.groupby(['response'])['sales_channel'].count().reset_index()
fig, ax = plt.subplots(figsize = (6,6))
ax.set_title('Total number of responses', fontweight='bold')
sns.barplot(x = response_channel['response'],y = response_channel['sales_channel'])
plt.xlabel("Response")
plt.ylabel("Freq.")

#%%
print("LOCATION\n", df['location_code'].value_counts(), "\n"*2, "EMPLOYMENT STAT.\n", df['employment_status'].value_counts(), "\n")
print("COVERAGE\n ", df['coverage'].value_counts(), "\n"*2, "VEHICLE CLASS\n",  df['vehicle_class'].value_counts())


#%%
## We checked for monthly_premium_auto, since it was correlated with the target variable (p = .6)
explanVars2 = df[['monthly_premium_auto', 'region', 'response', 'education', 'employment_status', 'gender', \
                  'location_code', 'policy_type', 'vehicle_class']]

fig, ax = plt.subplots(4,2, figsize = (10,15))
for i in range(1, len(explanVars2.columns) ):
    x1 = explanVars2.groupby(explanVars2.columns[i]).mean(['monthly_premium_auto'])
    plt.subplot(4,2,i)
    sns.barplot(x=x1.index, y='monthly_premium_auto', data=x1)
    plt.title(f'{explanVars2.columns[i]}')
    fig.subplots_adjust(hspace=1.5)


#%% Show a plot of the response by income.
df2 = df[['response', 'income']]
df2 = df2.groupby('response').mean()
fig = plt.figure(figsize=(8,6))
#ax.set_title('Income by response', fontweight='bold')
plt.title('Income by response', fontweight='bold')
sns.barplot(x = df2.index, y = df2['income'])


#%% (optional) Datetime format - Extract the months from the dataset and store in a separate column.
# Then filter the data to show only the information for the first quarter , ie. January, February and March.
# Hint: If data from March does not exist, consider only January and February.
df['month'] = df['effective_to_date'].str[0]
df['month'].replace(['1','2'],['Jan','Feb'])

#%%
def clean_gender(x):
    if x in ['M', 'Male']:
        return 'M'
    elif x in ["F", "female","Femal"]:
        return 'F'
    else:
        return 'U'

z = list(map(clean_gender, x))

#%%
def rrr(yy):
    x.replace({'a':'g', 'x':'k'})
    return yy
#%% Show a plot of the total number of responses.
fig, ax = plt.subplots(figsize = (6,6))
responses = df_cat['response'].value_counts()
ax.set_title('Total number of responses', fontweight='bold')
ax.bar(responses.index, responses, color=('khaki','lightblue'))

df_cat.columns
