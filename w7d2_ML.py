### LIBRARIES
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


oschdir = "C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories"
os.getcwd()

def lowerColumns(df):
    for i in df.columns:
        df.rename(columns = {f"{i}": f"{i.lower()}"}, inplace=True)

def Standardization(Series):
    mean_series = Series.mean()
    std_series = Series.std()
    Series = (Series - mean_series) / std_series
    return Series

def model_performance(y_train, y_pred_train, y_test, y_pred_test):

    ME_train = np.mean(np.exp(y_train)-np.exp(y_pred_train))
    ME_test  = np.mean(np.exp(y_test)-np.exp(y_pred_test))

    MAE_train = mean_absolute_error(np.exp(y_train),np.exp(y_pred_train))
    MAE_test  = mean_absolute_error(np.exp(y_test),np.exp(y_pred_test))

    MSE_train = mean_squared_error(np.exp(y_train),np.exp(y_pred_train))
    MSE_test  = mean_squared_error(np.exp(y_test),np.exp(y_pred_test))

    RMSE_train = np.sqrt(MSE_train)
    RMSE_test  = np.sqrt(MSE_test)

    MAPE_train = np.mean((np.abs(np.exp(y_train)-np.exp(y_pred_train)) / np.exp(y_train))* 100.)
    MAPE_test  = np.mean((np.abs(np.exp(y_test)-np.exp(y_pred_test)) / np.exp(y_test))* 100.)

    R2_train = r2_score(np.exp(y_train),np.exp(y_pred_train))
    R2_test  = r2_score(np.exp(y_test),np.exp(y_pred_test))

    performance = pd.DataFrame({'Error_metric': ['Mean error','Mean absolute error','Mean squared error',
                                                 'Root mean squared error','Mean absolute percentual error',
                                                 'R2'],
                                'Train': [ME_train, MAE_train, MSE_train, RMSE_train, MAPE_train, R2_train],
                                'Test' : [ME_test, MAE_test , MSE_test, RMSE_test, MAPE_test, R2_test]})

    pd.options.display.float_format = '{:.2f}'.format

    df_train = pd.DataFrame({'Real': np.exp(y_train), 'Predicted': np.exp(y_pred_train)})
    df_test  = pd.DataFrame({'Real': np.exp(y_test),  'Predicted': np.exp(y_pred_test)})

    return performance, df_train, df_test

def MSE_ols(y_tr, y_ts, x_tr, x_ts, n_neighbors):
    neigh = KNeighborsClassifier(n_neighbors).fit(x_tr, y_tr)
    y_pred_tr, y_pred_ts = neigh.predict(x_tr), neigh.predict(x_ts)
    y_tr = y_tr.squeeze()
    mse_tr = ((y_tr - y_pred_tr)**2).sum()/len(y_tr)
    mse_ts = ((y_ts - y_pred_ts)**2).sum()/len(y_ts)
    return mse_tr, mse_ts

def R2_ols(y_tr, y_ts, x_tr, x_ts, n_neighbors):
    neigh = KNeighborsClassifier(n_neighbors).fit(x_tr, y_tr)
    y_pred_tr, y_pred_ts = neigh.predict(x_tr), neigh.predict(x_ts)
    y_tr, y_ts, y_pred_ts = y_tr.squeeze(), y_ts.squeeze(), y_pred_ts.squeeze()
    r2_tr = 1 - ((y_tr - y_pred_tr)**2).sum() / ((y_tr - y_tr.mean())**2).sum()
    r2_ts = 1 - ((y_ts - y_pred_ts)**2).sum() / ((y_ts - y_ts.mean())**2).sum()
    return r2_tr, r2_ts



#%% donwload and clean data
df = pd.read_csv("./data/labs/labML.csv")
lowerColumns(df)
df = df[['tenure', 'seniorcitizen', 'monthlycharges', 'totalcharges', 'churn']]
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df['totalcharges'].isnull().sum()
df = df.dropna()
df['churn'] = df['churn'].replace({"No":0, "Yes":1})


#%%
x = df[['tenure', 'seniorcitizen', 'monthlycharges', 'totalcharges']]
x = x.astype(float)
y = df['churn']

y_tr, y_ts, x_tr, x_ts = train_test_split(y, x, test_size=.3, random_state=123)

#%%
# for loop to try many values of k
full = pd.DataFrame()
models = {'k': [] }

for k in range(2,21):

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_tr, y_tr)

    models['k'] = [k, neigh]

    y_pred_train_knn = neigh.predict(x_tr)
    y_pred_test_knn  = neigh.predict(x_ts)

    performance_knn, _, _ = model_performance(y_tr, y_pred_train_knn, y_ts, y_pred_test_knn)
    temp = pd.DataFrame({'k': [k]*6, 'Error_metric': performance_knn['Error_metric'],
                         'Train': performance_knn['Train'], 'Test': performance_knn['Test']})
    full = pd.concat([full,temp], axis=0)

full2 = full.melt(id_vars=['k','Error_metric'])
full2
#%%
#metrics = ['Mean error',]'Mean absolute error',...]
fig, ax = plt.subplots(2,3, figsize=(20,10))
sns.lineplot(x = 'k', y = 'value', data = full2[full2['Error_metric'] == 'Mean error'],
             hue = 'variable', ax = ax[0,0])
ax[0,0].set_xticks(range(2,21))
ax[0,0].set_title("Mean error")
ax[0,0].legend(loc='lower right')
sns.lineplot(x = 'k', y = 'value', data = full2[full2['Error_metric'] == 'Mean absolute error'],
             hue = 'variable', ax = ax[0,1])
ax[0,1].set_xticks(range(2,21))
ax[0,1].set_title("Mean absolute error")
ax[0,1].legend(loc='lower right')
sns.lineplot(x = 'k', y = 'value', data = full2[full2['Error_metric'] == 'Mean squared error'],
             hue = 'variable', ax = ax[0,2])
ax[0,2].set_xticks(range(2,21))
ax[0,2].set_title("Mean squared error")
ax[0,2].legend(loc='lower right')
sns.lineplot(x = 'k', y = 'value', data = full2[full2['Error_metric'] == 'Root mean squared error'],
             hue = 'variable', ax = ax[1,0])
ax[1,0].set_xticks(range(2,21))
ax[1,0].set_title("Root mean squared error")
ax[1,0].legend(loc='lower right')
sns.lineplot(x = 'k', y = 'value', data = full2[full2['Error_metric'] == 'Mean absolute percentual error'],
             hue = 'variable', ax = ax[1,1])
ax[1,1].set_xticks(range(2,21))
ax[1,1].set_title("Mean absolute percentual error")
ax[1,1].legend(loc='lower right')
sns.lineplot(x = 'k', y = 'value', data = full2[full2['Error_metric'] == 'R2'],
             hue = 'variable', ax = ax[1,2])
ax[1,2].set_xticks(range(2,21))
ax[1,2].set_title("R2")
ax[1,2].legend(loc='lower right')


#%% NOW FIT THE MODEL
neigh = KNeighborsClassifier(n_neighbors=15) # n_neighbors = K
neigh.fit(x_tr, y_tr) # Minkowski distance with p = 2 -> Euclidean distance

y_pred_train_knn = neigh.predict(x_tr)
y_pred_test_knn  = neigh.predict(x_ts)

MSE_ols(y_tr, y_ts, x_tr, x_ts, 12)
R2_ols(y_tr, y_ts, x_tr, x_ts, 12)


performance_knn, _, _ = model_performance(y_tr, y_pred_train_knn, y_ts, y_pred_test_knn)
performance_knn


#%%
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuPu"
plot_confusion_matrix(neigh, x_tr, y_tr,ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")

plot_confusion_matrix(neigh, x_ts, y_ts,ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
y_pred = neigh.predict(x_ts)

def evaluate_classification_model(y_train, y_pred_train, y_test, y_pred_test):
    performance_df = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                                   'Train': [accuracy_score(y_train, y_pred_train),
                                             precision_score(y_train, y_pred_train),
                                             recall_score(y_train, y_pred_train)],
                                   'Test': [accuracy_score(y_test, y_pred_test),
                                            precision_score(y_test, y_pred_test),
                                            recall_score(y_test, y_pred_test)]})

    pd.options.display.float_format = '{:.2f}'.format

    df_train = pd.DataFrame({'Real': y_train, 'Predicted': y_pred_train})
    df_test  = pd.DataFrame({'Real': y_test,  'Predicted': y_pred_test})

    return performance_df, df_train, df_test

## calling the function
error_metrics_df,y_train_vs_predicted, \
y_test_vs_predicted=evaluate_classification_model(y_tr, y_pred_train_knn,
                                                  y_ts, y_pred)
error_metrics_df



#%% RANDOM FOREST
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.30, random_state=11)

# Bear in mind that sklearn uses a different function for decission trees used for
# classification ( to predict a categorical feature ): DecisionTreeClassifier()
trees = DecisionTreeClassifier(max_depth=3)

trees.fit(x_tr, y_tr)

y_pred_train_dt = trees.predict(x_tr)
y_pred_test_dt = trees.predict(x_ts)

performance_df = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                               'Train': [accuracy_score(y_tr, y_pred_train_dt),
                                         precision_score(y_tr, y_pred_train_dt),
                                         recall_score(y_tr, y_pred_train_dt)],
                               'Test': [accuracy_score(y_ts, y_pred_test_dt),
                                        precision_score(y_ts, y_pred_test_dt),
                                        recall_score(y_ts, y_pred_test_dt)]})

fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuPu"

#print("Confusion matrix for the train set")
#print(confusion_matrix(y_train,y_pred_train_dt).T)
plot_confusion_matrix(trees, x_tr, y_tr, ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")
#print(confusion_matrix(y_test,y_pred_test_dt).T)
plot_confusion_matrix(trees, x_ts, y_ts,ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")

error_metrics_df,y_train_vs_predicted, y_test_vs_predicted = evaluate_classification_model(y_tr, y_pred_train_knn, y_ts, y_pred)
error_metrics_df

#%%
plt.figure(figsize=(10,7))
plt.barh(x.columns, trees.feature_importances_)


#%%
from sklearn.tree import plot_tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (34,20))
plot_tree(trees,filled = True, rounded=True,feature_names=x.columns)
plt.show()


#%% K FOLD
# apply K-fold cross validation on your models before and check the model score
set(y)
scores = cross_val_score(trees, x_tr, y_tr, cv=5)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
y_pred_tr = cross_val_predict(trees, x_tr, y_tr, cv=5)
y_pred_ts = cross_val_predict(trees, x_ts, y_ts, cv=5)

y_ts.mean()
y_pred.mean()

error_metrics_df, y_train_vs_predicted, y_test_vs_predicted = \
    evaluate_classification_model(y_tr, y_pred_tr, y_ts, y_pred_ts)
print(error_metrics_df)

#print(confusion_matrix(y_test,y_pred_test_dt).T)
fig, ax = plt.subplots(1,2, figsize=(14,8))
color_map = "BuPu"

plot_confusion_matrix(trees, x_tr, y_tr, ax=ax[0], values_format = 'd', cmap=color_map)
ax[0].title.set_text("Train Set")
plot_confusion_matrix(trees, x_ts, y_ts, ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")


#%%
tree = DecisionTreeClassifier()
ols = LinearRegression()
knn = KNeighborsRegressor()

model_pipeline = [tree, ols, knn]
model_names = ['Regression Tree', 'Linear Regression', 'KNN']
scores_tr, scores_ts = {}, {}
i=0
for model in model_pipeline:
    mean_score_tr = np.mean(cross_val_score(model, x_tr, y_tr, cv=5))
    scores_tr[model_names[i]] = mean_score_tr
    mean_score_ts = np.mean(cross_val_score(model, x_ts, y_ts, cv=5))
    scores_ts[model_names[i]] = mean_score_ts
    i = i+1

scores_df_tr = pd.DataFrame.from_dict(scores_tr, orient='index')
scores_df_ts = pd.DataFrame.from_dict(scores_ts, orient='index')


#%%
plt.figure(figsize=(10,15))
plt.subplot(4,2,1)
plt.bar(scores_df_tr.index, scores_df_tr.iloc[:,0])
plt.title("Train")
plt.subplot(4,2,2)
plt.bar(scores_df_ts.index, scores_df_ts.iloc[:,0])
plt.title("Test")


#%% Managing imbalance in the dataset
# Use the resampling strategies used in class for upsampling and downsampling
# to create a balance between the two classes.
df['churn'].value_counts()


#%% ## Downsampling
df['churn'] = df['churn'].replace({"No":0, "Yes":1})

category_0 = df[df['churn'] == 0]
category_1 = df[df['churn'] == 1]

category_0_down = category_0.sample(len(category_1))
df_down1 = pd.concat([category_0_down, category_1], axis=0)
df_down1 = df_down1.sample(frac=1)
df_down1.value_counts()
df_down1['churn'].value_counts()


#%% Downsampling with Tomelinks
from imblearn.under_sampling import TomekLinks
tl = TomekLinks('majority')
x = df.drop(['churn'], axis=1)
y = df['churn']
x, y = tl.fit_resample(x,y)
y.value_counts()
df_tome = pd.merge(x, y, right_index=True, left_index=True)
df_tome['churn'].value_counts()


#%% Upsampling 1
category_0 = df[df['churn'] == 0]
category_1 = df[df['churn'] == 1]
category_1_up = category_1.sample(len(category_0), replace=True)
print(category_1_up.shape)
df_up1 = pd.concat([category_0, category_1_up], axis=0)
df_up1 = df_up1.sample(frac=1)     #shuffling the data
df_up1['churn'].value_counts()


#%% Upsampling with SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x = df.drop(['churn'], axis=1)
y = df['churn']
x_sm, y_sm = sm.fit_resample(x, y)
df_smote = pd.merge(x_sm, y_sm, right_index=True, left_index=True)
df_smote['churn'].value_counts()


#%%
def ApplyTrees(df):
    y = df['churn']
    x = df.drop(['churn'], axis=1)

    x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.30, random_state=11)
    trees = DecisionTreeClassifier(max_depth=3)

    trees.fit(x_tr, y_tr)

    y_pred_train_dt = trees.predict(x_tr)
    y_pred_test_dt = trees.predict(x_ts)

    performance_df = pd.DataFrame({'Error_metric': ['Accuracy','Precision','Recall'],
                                   'Train': [accuracy_score(y_tr, y_pred_train_dt),
                                             precision_score(y_tr, y_pred_train_dt),
                                             recall_score(y_tr, y_pred_train_dt)],
                                   'Test': [accuracy_score(y_ts, y_pred_test_dt),
                                            precision_score(y_ts, y_pred_test_dt),
                                            recall_score(y_ts, y_pred_test_dt)]})

    return performance_df

#%%
df_list = [df_down1, df_tome, df_up1, df_smote]
l2 = []
for i in df_list:
    l = ApplyTrees(i)
    l2.append(l)

ll = l2[0]
ll
#%%

fig=plt.figure(figsize=(10,7))
plt.subplot(2,2,1)
plt.bar(ll['Error_metric'], ll['Train'])
plt.title("Train")
plt.subplot(2,2,2)
plt.bar(ll['Error_metric'], ll['Test'])
plt.title("Test")
fig.suptitle('Sub sampling 1')
plt.show()

#%%
def PlotMetrics(l2):
    ## training
    fig=plt.figure(figsize=(10,4))
    plt.subplot(1,4,1)
    plt.bar(l2[0]['Error_metric'], l2[0]['Train'])
    plt.title("Subsampling 1 ")
    plt.xticks(rotation=45)
    plt.subplot(1,4,2)
    plt.bar(l2[1]['Error_metric'], l2[1]['Train'])
    plt.title("Sub - Tomelink")
    plt.xticks(rotation=45)
    plt.subplot(1,4,3)
    plt.bar(l2[2]['Error_metric'], l2[2]['Train'])
    plt.title("Upsampling 1")
    plt.xticks(rotation=45)
    plt.subplot(1,4,4)
    plt.bar(l2[3]['Error_metric'], l2[3]['Train'])
    plt.title("Upsampling - SMOTE")
    plt.xticks(rotation=45)
    plt.ylim(bottom=.4, top=.9)
    fig.suptitle('TRAINING')
    plt.show()

    ## test
    fig=plt.figure(figsize=(10,4))
    plt.subplot(1,4,1)
    plt.bar(l2[0]['Error_metric'], l2[0]['Test'])
    plt.title("Subsampling 1 ")
    plt.xticks(rotation=45)
    plt.subplot(1,4,2)
    plt.bar(l2[1]['Error_metric'], l2[1]['Test'])
    plt.title("Sub - Tomelink")
    plt.xticks(rotation=45)
    plt.subplot(1,4,3)
    plt.bar(l2[2]['Error_metric'], l2[2]['Test'])
    plt.title("Upsampling 1")
    plt.xticks(rotation=45)
    plt.subplot(1,4,4)
    plt.bar(l2[3]['Error_metric'], l2[3]['Test'])
    plt.title("Upsampling - SMOTE")
    plt.xticks(rotation=45)
    plt.ylim(bottom=.4, top=.9)
    fig.suptitle('TEST')
    plt.show()

#%%
PlotMetrics(l2)

'''
l2[0]['Error_metric'] =

    fig, ax = plt.subplots(1,2, figsize=(14,8))
    color_map = "BuPu"

    plot_confusion_matrix(trees, x_tr, y_tr, ax=ax[0], values_format = 'd', cmap=color_map)
    ax[0].title.set_text("Train Set")

    plot_confusion_matrix(trees, x_ts, y_ts,ax=ax[1],values_format = 'd', cmap=color_map)
    ax[1].title.set_text("Test Set")

    print(len(x_tr), len(y_tr), len(y_pred_train_dt), len(x_ts), len(y_ts), len(y_pred_test_dt))

    return error_metrics_df
'''

#%% fit a Random forest Classifier on the data and compare the accuracy.
# use sub tomelink df
df
x = df.drop(['churn'], axis=1)
x_num = df[['tenure', 'monthlycharges', 'totalcharges']]
x_num = Standardization(x_num)
x_cat = df['seniorcitizen']
x = pd.merge(x_num, x_cat, left_index=True, right_index=True)
y = df['churn']

x = df[['tenure', 'seniorcitizen', 'monthlycharges', 'totalcharges']]


#%% train the model
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.30, random_state=11)

clf = RandomForestClassifier(max_depth=6,min_samples_leaf=20,max_features=None,n_estimators=100,
                             bootstrap=True,oob_score=True, random_state=0)
clf.fit(x_tr, y_tr)
print(clf.score(x_tr, y_tr))
print(clf.score(x_ts, y_ts))


#%% CROSS VALIDATION ON RF
clf = RandomForestClassifier(max_depth=3,min_samples_leaf=20,max_features=None,n_estimators=100,
                             bootstrap=True,oob_score=True, random_state=0)
cross_val_scores = cross_val_score(clf, x_tr, y_tr, cv=5)
cross_val_scores
np.std(cross_val_scores)


#%% tune the hyper paramters with gridsearch and check the results.
param_grid = {
    'n_estimators': [50, 100, 500],
    'min_samples_split': [2, 4],
    'min_samples_leaf' : [1, 2],
    'max_features': ['sqrt']
    ##'max_samples' : ['None', 0.5],
    ##'max_depth':[3,5,10],
    ## 'bootstrap':[True,False]
}

grid_search = GridSearchCV(clf, param_grid, cv=5,return_train_score=True,n_jobs=-1,)


#%%  show the best parameters
grid_search.fit(x_tr, y_tr)
grid_search.best_params_   #To check the best set of parameters returned
results = pd.DataFrame(grid_search.cv_results_)

## use the parameters in the RF
clf = RandomForestClassifier(random_state=12345, max_features='sqrt',
                             min_samples_leaf=1, min_samples_split=2, n_estimators=500)
cross_val_scores = cross_val_score(clf, x_tr, y_tr, cv=10)
print(np.mean(cross_val_scores))
print(np.std(cross_val_scores))


#%% FEATURE SELECTION - higher the score, higher the importance
clf.fit(x_tr, y_tr)
len(x_tr.columns)
feature_names = x_tr.columns
feature_names = list(feature_names)
results_imp = pd.DataFrame(list(zip(feature_names, clf.feature_importances_)))
results_imp.columns = ['columns_name', 'score_feature_importance']
results_imp.sort_values(by=['score_feature_importance'], ascending = False)


