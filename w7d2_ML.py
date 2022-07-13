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



#%%
df = pd.read_csv("./data/labs/labML.csv")
lowerColumns(df)
df = df[['tenure', 'seniorcitizen', 'monthlycharges', 'totalcharges', 'churn']]
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df['totalcharges'].isnull().sum()
df = df.dropna()
x = df[['tenure', 'seniorcitizen', 'monthlycharges', 'totalcharges']]
x = x.astype(float)
y = df['churn']

y = y.replace({"No":0, "Yes":1})

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
neigh = KNeighborsClassifier(n_neighbors=12) # n_neighbors = K
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

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
plot_confusion_matrix(trees, x_tr, y_tr,ax=ax[1],values_format = 'd', cmap=color_map)
ax[1].title.set_text("Test Set")


#%%
plt.figure(figsize=(20,14))
plt.barh(x.columns, trees.feature_importances_)

#%%
from sklearn.tree import plot_tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (34,20))
plot_tree(trees,filled = True, rounded=True,feature_names=x.columns)
plt.show()
#%%
