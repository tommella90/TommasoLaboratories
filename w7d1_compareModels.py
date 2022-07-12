###MODEL COMPARISON
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


## LOWER CASE FOR ALL COLUMNS NAMES
def lowerColumns(df):
    for i in df.columns:
        df.rename(columns = {f"{i}": f"{i.lower()}"}, inplace=True)

def Standardization(Series):
    mean_series = Series.mean()
    std_series = Series.std()
    Series = (Series - mean_series) / std_series
    return Series

def SplitScaleData(y, x, ratio, cat_len):
    y_tr, y_ts, x_tr, x_ts = train_test_split(y, x, test_size=ratio, random_state=123)
    xtrz1 = Standardization(x_tr.iloc[:, 0: cat_len])
    xtrz2 = x_tr.iloc[:, cat_len:]
    x_tr_z = xtrz1.merge(xtrz2, left_index=True, right_index=True)
    xtsz1 = Standardization(x_ts.iloc[:,0:cat_len])
    xtsz2 = x_ts.iloc[:, cat_len:]
    x_ts_z = xtsz1.merge(xtsz2, left_index=True, right_index=True)
    return y_tr, y_ts, x_tr_z, x_ts_z

def LinearFit(y,x):
    x = sm.add_constant(x)
    ols = sm.OLS(y, x).fit()
    ols_table = ols.summary()
    return ols_table

def TableResults(ols_table):
    ## format the ols table
    table = pd.DataFrame(ols_table.tables[1])
    list_col = ['ind_var', 'coef', 'std_err', 't_val', 'p_val', 'low_ci', 'high_ci']
    for num in range(0,len(table.columns)):
        table.rename(columns={table.columns[num]: f"{list_col[num]}" }, inplace = True)
    table = table.iloc[1:,:]
    #ols = sm.OLS(y, x).fit()
    #table['obs'] = ols.nobs
    table.rename(columns={"P>|t|": "p_val", "[0.025": "low_ci", "0.975]": "high_ci"}, inplace=True)
    table = table.set_index(table.columns[0])
    table = round(table, 3)
    table1 = table.applymap(str).applymap(float)
    return table1

def TableResults2(result):
    table = pd.DataFrame(result.tables[0].data)
    table = table.iloc[:,2:4].T
    table.columns = table.iloc[0]
    table2 = table.iloc[1:,]
    return table2

def MSE_ols(y_tr, y_ts, x_tr, x_ts):
    ols = sm.OLS(y_tr, x_tr).fit()
    y_pred_tr, y_pred_ts = ols.predict(x_tr), ols.predict(x_ts)
    y_tr = y_tr.squeeze()
    mse_tr = ((y_tr - y_pred_tr)**2).sum()/len(y_tr)
    mse_ts = ((y_ts - y_pred_ts)**2).sum()/len(y_ts)
    return mse_tr, mse_ts

def R2_ols(y_tr, y_ts, x_tr, x_ts):
    ols = sm.OLS(y_tr, x_tr).fit()
    y_pred_tr, y_pred_ts = ols.predict(x_tr), ols.predict(x_ts)
    y_tr, y_ts, y_pred_ts = y_tr.squeeze(), y_ts.squeeze(), y_pred_ts.squeeze()
    r2_tr = 1 - ((y_tr - y_pred_tr)**2).sum() / ((y_tr - y_tr.mean())**2).sum()
    r2_ts = 1 - ((y_ts - y_pred_ts)**2).sum() / ((y_ts - y_ts.mean())**2).sum()
    return r2_tr, r2_ts

def MSE_lasso(y_tr, y_ts, x_tr, x_ts):
    lasso = Lasso(alpha=0.05)
    lasso.fit(x_tr, y_tr)
    y_pred_tr, y_pred_ts = lasso.predict(x_tr), lasso.predict(x_ts)
    y_tr = y_tr.squeeze()
    mse_tr = ((y_tr - y_pred_tr)**2).sum()/len(y_tr)
    mse_ts = ((y_ts - y_pred_ts)**2).sum()/len(y_ts)
    return mse_tr, mse_ts

def R2_lasso(y_tr, y_ts, x_tr, x_ts):
    lasso = Lasso(alpha=0.05)
    lasso.fit(x_tr, y_tr)
    y_pred_tr, y_pred_ts = lasso.predict(x_tr), lasso.predict(x_ts)
    y_tr, y_ts, y_pred_ts = y_tr.squeeze(), y_ts.squeeze(), y_pred_ts.squeeze()
    r2_tr = 1 - ((y_tr - y_pred_tr)**2).sum() / ((y_tr - y_tr.mean())**2).sum()
    r2_ts = 1 - ((y_ts - y_pred_ts)**2).sum() / ((y_ts - y_ts.mean())**2).sum()
    return r2_tr, r2_ts

def MSE_ridge(y_tr, y_ts, x_tr, x_ts):
    ridge = Ridge(alpha=100)
    ridge.fit(x_tr, y_tr)
    y_pred_tr, y_pred_ts = ridge.predict(x_tr), ridge.predict(x_ts)
    y_tr = y_tr.squeeze()
    mse_tr = ((y_tr - y_pred_tr)**2).sum()/len(y_tr)
    mse_ts = ((y_ts - y_pred_ts)**2).sum()/len(y_ts)
    return mse_tr, mse_ts

def R2_ridge(y_tr, y_ts, x_tr, x_ts):
    ridge = Ridge(alpha=100)
    ridge.fit(x_tr, y_tr)
    y_pred_tr, y_pred_ts = ridge.predict(x_tr), ridge.predict(x_ts)
    y_tr, y_ts, y_pred_ts = y_tr.squeeze(), y_ts.squeeze(), y_pred_ts.squeeze()
    r2_tr = 1 - ((y_tr - y_pred_tr)**2).sum() / ((y_tr - y_tr.mean())**2).sum()
    r2_ts = 1 - ((y_ts - y_pred_ts)**2).sum() / ((y_ts - y_ts.mean())**2).sum()
    return r2_tr, r2_ts


#%%
def CompareModels(x, y):
    ## we start from a clean dataframe, splitted in x(ind.vars) and y (target)
    y_tr, y_ts, x_tr, x_ts = train_test_split(y, x, test_size=.3, random_state=123)

    ## get OLS results
    ols = sm.OLS(y_tr, x_tr).fit()
    mse_ols_tr, mse_ols_ts = MSE_ols(y_tr, y_ts, x_tr, x_ts)
    r2_ols_tr, r2_ols_ts = R2_ols(y_tr, y_ts, x_tr, x_ts)

    ## get lasso results
    lasso = Lasso(alpha=0.05)
    lasso.fit(x_tr, y_tr)
    r2_lasso_tr, r2_lasso_ts = R2_lasso(y_tr, y_ts, x_tr, x_ts)
    mse_lasso_tr, mse_lasso_ts = MSE_lasso(y_tr, y_ts, x_tr, x_ts)

    ## get ridge results
    ridge = Ridge(alpha=.5)
    ridge.fit(x_tr, y_tr)
    r2_ridge_tr, r2_ridge_ts = R2_ridge(y_tr, y_ts, x_tr, x_ts)
    mse_ridge_tr, mse_ridge_ts = MSE_ridge(y_tr, y_ts, x_tr, x_ts)

    model_stats = {
        "r2_ols_train": r2_ols_tr,
        "r2_ols_test": r2_ols_ts,
        "r2_lasso_train": r2_lasso_tr,
        "r2_lasso_test": r2_lasso_ts,
        "r2_ridge_train": r2_ridge_tr,
        "r2_ridge_test": r2_ridge_ts,

        "mse_ols_train": mse_ols_tr,
        "mse_ols_test": mse_ols_tr,
        "mse_lasso_train": mse_lasso_tr,
        "mse_lasso_test": mse_lasso_tr,
        "mse_ridge_train": mse_ridge_tr,
        "mse_ridge_test": mse_ridge_tr
    }
    results = pd.DataFrame.from_dict(model_stats, orient='index', columns=['values'])
    return results

#%%
def CompareFilterModels(x,y):
    ## split test train set
    y_tr, y_ts, x_tr, x_ts = train_test_split(y, x, test_size=.3, random_state=123)

    ## p value filtering
    ols_table = LinearFit(y_tr, x_tr)
    results_as_html = ols_table.tables[1].as_html()
    table2 = pd.read_html(results_as_html, header=0, index_col=0)[0]
    ind_vars = list(table2[table2['t'] >= 1.96].index)
    x_filter_p = x.filter(ind_vars)

    ## RFE filtering
    ols = LinearRegression()
    selector = RFE(ols, n_features_to_select= 9, step = 1, verbose = 1)
    selector.fit(x_tr, y_tr)
    kept_features = selector.get_support(indices = True)
    kept_features = list(x_tr.iloc[:,kept_features].columns)
    x_rfe = x.filter(kept_features)

    basic_model = CompareModels(x,y)
    basic_model.rename(columns={'0': 'booo'})
    pval_filter = CompareModels(x_filter_p, y)
    rfe_filter = CompareModels(x_rfe, y)

    all_models = basic_model.merge(pval_filter, left_index=True, right_index=True,
                                   suffixes=('_all', '_pval_filter'))
    all_models = all_models.merge(rfe_filter, left_index=True, right_index=True)
    all_models.rename(columns={'values': 'values_rfe_filter'}, inplace=True)

    return all_models


#%%
def PlotsModelComparison(table_res):

    titles = ['NO FILTERING', 'P VALUE SELECTION', 'RFE SELECTION']

    names = list(table_res.index)[0:6]
    fig, ax = plt.subplots(1,3, figsize = (13,4))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        all_r2 = table_res.iloc[0:6,i]
        list_color = ['r', 'r', 'g', 'g', 'orange', 'orange']
        plt.ylim(bottom=0.73, top=0.78)
        plt.xticks(rotation=45)
        plt.title(f"R2: {titles[i]}")
        plt.bar(names, all_r2, color=list_color, label=titles)

    names = list(table_res.index)[6:]
    fig, ax = plt.subplots(1,3, figsize = (13,4))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        all_mse2 = table_res.iloc[6:, i]
        list_color = ['r', 'r', 'g', 'g', 'orange', 'orange']
        plt.ylim(bottom=19000, top=23000)
        plt.xticks(rotation=45)
        plt.title(f"MSE: {titles[i]}")
        plt.bar(names, all_mse2, color=list_color)


#%%
df = pd.read_csv("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/data/labs/Data_Marketing_Customer_Analysis_Round3.csv")
df = df.iloc[:, 1:]

## drop: effective_to_date'
y = df['total_claim_amount']

df_cat = df[['region', 'response', 'coverage',
             'education', 'month', 'employment_status',
             'gender', 'location_code', 'marital_status', 'number_of_open_complaints',
             'number_of_policies', 'policy_type', 'policy', 'renew_offer_type',
             'sales_channel', 'vehicle_class', 'vehicle_size']]
df_cat = pd.get_dummies(df_cat, drop_first=True)

df_num = df[['customer_lifetime_value', 'income',
             'monthly_premium_auto', 'months_since_last_claim',
             'months_since_policy_inception']]
df_num = Standardization(df_num)

x = df_cat.merge(df_num, left_index=True, right_index=True)


#%%
table_res = CompareFilterModels(x, y)
table_res
PlotsModelComparison(table_res)




#%%

#%%
import matplotlib.pyplot as plt


titles = ['OLS', 'P VALUE SELECTION', 'RFE SELECTION']

names = list(table_res.index)[6:]

fig, ax = plt.subplots(1,3, figsize = (13,4))
for i in range(0,3):
    plt.subplot(1,3,i+1)
    all_r2 = table_res.iloc[0:6,i]
    list_color = ['r', 'r', 'g', 'g', 'orange', 'orange']
    plt.ylim(bottom=0.73, top=0.78)
    plt.xticks(rotation=45)
    plt.title(f"R2: {titles[i]}")
    plt.bar(names, all_r2, color=list_color)

fig, ax = plt.subplots(1,3, figsize = (13,4))
for i in range(0,3):
    plt.subplot(1,3,i+1)
    all_mse2 = table_res.iloc[6:, i]
    list_color = ['r', 'r', 'g', 'g', 'orange', 'orange']
    plt.ylim(bottom=19000, top=23000)
    plt.xticks(rotation=45)
    plt.title(f"MSE: {titles[i]}")
    plt.bar(names, all_mse2, color=list_color)


#%%


def PlotsModelComparison(table_res):

    titles = ['OLS', 'P VALUE SELECTION', 'RFE SELECTION']
    names = list(table_res.index)[6:]

    fig, ax = plt.subplots(1,3, figsize = (13,4))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        all_r2 = table_res.iloc[0:6,i]
        list_color = ['r', 'r', 'g', 'g', 'orange', 'orange']
        plt.ylim(bottom=0.73, top=0.78)
        plt.xticks(rotation=45)
        plt.title(f"R2: {titles[i]}")
        plt.bar(names, all_r2, color=list_color)

    fig, ax = plt.subplots(1,3, figsize = (13,4))
    for i in range(0,3):
        plt.subplot(1,3,i+1)
        all_mse2 = table_res.iloc[6:, i]
        list_color = ['r', 'r', 'g', 'g', 'orange', 'orange']
        plt.ylim(bottom=19000, top=23000)
        plt.xticks(rotation=45)
        plt.title(f"MSE: {titles[i]}")
        plt.bar(names, all_mse2, color=list_color)
#%%

## split test and train
y_tr, y_ts, x_tr, x_ts = train_test_split(y, x, test_size=.3, random_state=123)

#%% 1) OLS
ols = sm.OLS(y_tr, x_tr).fit()
mse_ols_tr, mse_ols_ts = MSE_ols(y_tr, y_ts, x_tr, x_ts)
r2_ols_tr, r2_ols_ts = R2_ols(y_tr, y_ts, x_tr, x_ts)

print(mse_ols_tr, mse_ols_ts, r2_ols_tr, r2_ols_ts)


#%% LASSO
lasso = Lasso(alpha=0.05)
lasso.fit(x_tr, y_tr)
r2_lasso_tr, r2_lasso_ts = R2_lasso(y_tr, y_ts, x_tr, x_ts)
mse_lasso_tr, mse_lasso_ts = MSE_lasso(y_tr, y_ts, x_tr, x_ts)

print(mse_lasso_tr, mse_lasso_ts, r2_lasso_tr, r2_lasso_ts)


#%% RIDGE
ridge = Ridge(alpha=100)
ridge.fit(x_tr, y_tr)
r2_ridge_tr, r2_ridge_ts = R2_ridge(y_tr, y_ts, x_tr, x_ts)
mse_ridge_tr, mse_ridge_ts = MSE_ridge(y_tr, y_ts, x_tr, x_ts)

print(mse_ridge_tr, mse_ridge_ts, r2_ridge_tr, r2_ridge_ts)


#%% P VALUE FILTERING
ols_table = LinearFit(y_tr, x_tr)
results_as_html = ols_table.tables[1].as_html()
table2 = pd.read_html(results_as_html, header=0, index_col=0)[0]
ind_vars = list(table2[table2['P>|t|'] <=0.05].index)

x_filter_p = x.filter(ind_vars)

y_tr, y_ts, x_tr, x_ts = train_test_split(y, x_filter_p, test_size=.3, random_state=123)
ols = sm.OLS(y_tr, x_tr).fit()
mse_pval_tr, mse_pval_ts = MSE_ols(y_tr, y_ts, x_tr, x_ts)
r2_pval_tr, r2_pval_ts = R2_ols(y_tr, y_ts, x_tr, x_ts)

print(len(x_filter_p))
print(mse_pval_tr, mse_pval_ts, r2_pval_tr, r2_pval_ts)


#%% RECURSIVE FEATURE EILIMINATION
ols = LinearRegression()
selector = RFE(ols, n_features_to_select= 15, step = 1, verbose = 1)
selector.fit(x_tr, y_tr)

kept_features = selector.get_support(indices = True)
kept_features = list(x_tr.iloc[:,kept_features].columns)

x_rfe = x.filter(kept_features)
y_tr, y_ts, x_tr, x_ts = train_test_split(y, x_rfe, test_size=.3, random_state=123)
ols = sm.OLS(y_tr, x_tr).fit()
mse_rfe_tr, mse_rfe_ts = MSE_ols(y_tr, y_ts, x_tr, x_ts)
r2_rfe_tr, r2_rfe_ts = R2_ols(y_tr, y_ts, x_tr, x_ts)

print(len(kept_features))
print(mse_rfe_tr, mse_rfe_ts, r2_rfe_tr, r2_rfe_ts)


#%%
model_stats = {
    "r2_ols_train": r2_ols_tr,
    "r2_ols_test": r2_ols_ts,
    "mse_ols_train": mse_ols_tr,
    "mse_ols_test": mse_ols_tr,
    "r2_lasso_train": r2_lasso_tr,
    "r2_lasso_test": r2_lasso_ts,
    "mse_lasso_train": mse_lasso_tr,
    "mse_lasso_test": mse_lasso_tr,
    "r2_ridge_train": r2_ridge_tr,
    "r2_ridge_test": r2_ridge_ts,
    "mse_ridge_train": mse_ridge_tr,
    "mse_ridge_test": mse_ridge_tr,
    "r2_pval_train": r2_pval_tr,
    "r2_pval_test": r2_pval_ts,
    "mse_pval_train": mse_pval_tr,
    "mse_pval_test": mse_pval_tr,
    "r2_rfe_train": r2_rfe_tr,
    "r2_rfe_test": r2_rfe_ts,
    "mse_rfe_train": mse_rfe_tr,
    "mse_rfe_test": mse_rfe_tr
}

results = pd.DataFrame.from_dict(model_stats, orient='index', columns=['values'])
results