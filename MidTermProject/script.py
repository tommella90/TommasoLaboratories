###############################
## IMPLICIT RACE ASSOCIATION ##
###############################
### IMPORT LIBRARIES
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
os.chdir("C:/Users/tomma/Documents/data_science/berlin/projects/data")

###1) DATA CLEANING
def CLEANING_DATA(df):

    def lowerColumns(dataframe):
        for i in dataframe.columns:
            dataframe.rename(columns = {f"{i}": f"{i.lower()}"}, inplace=True)

    def sort_missing(dataframe):
        missing, keys, values = {}, [], []
        for i in dataframe.columns:
            keys.append(i)
            values.append(dataframe[f'{i}'].isna().sum())
        missing = dict(zip(keys, values))
        missing = pd.DataFrame(missing.values(), missing.keys())
        missing = missing.iloc[1:, :]
        missing.rename(columns = {0: "missing"}, inplace=True)
        missing.sort_values(by="missing", ascending=True, inplace=True)
        missing.reset_index(inplace=True)
        return missing

    def find_mean(subset, sample):
        newcol = subset.sum(axis=1) / sample
        return newcol

    ## data
    #df = pd.read_stata('iat_sex2020_clean.dta', convert_categoricals=False)
    lowerColumns(df)
    df = df.dropna()
    df = df.loc[df['num_002'] == 1]
    df.drop(['num_002'], axis = 1, inplace = True)

    df['gender_feel'] = find_mean(df[['tgayleswomen', 'tgaymen', 'tstraightmen', 'tstraightwomen']], 4)
    df['gender_preg'] = find_mean(df[['adoptchild', 'marriagerights_3num', 'relationslegal_3num',
                                      'serverights', 'transgender', 'countrycit_num']], 6)


    ## categorical vars
    df_cat = df[['birthsex', 'genderidentity', 'sexuality_5', 'raceomb_002',
                 'contactfamily_num', 'contactfriend_num', 'side_straight_34',
                 'contactfriendly_num', 'contactmet_num']]

    df_cat.rename(columns = {"birthsex": "sex",
                             "genderidentity": "gn_id",
                             "sexuality_5": "straight",
                             "raceomb_002": "race",
                             "side_straight_34": "straight_first",
                             "contactfamily_num": "fam_gay",
                             "contactfriend_num": "friend_gay",
                             "contactfriendly_num": "friendly",
                             "contactmet_num": "met_gay"},
                  inplace=True, errors='raise')


    ## numerical vars
    df_num = df[['_v1', 'birthyear', 'gender_preg', 'gender_feel', 'att_7', 'edu_14', 'politicalid_7', 'religionid']]
    df_num.rename(columns = {"_v1": "iat",
                             "birthyear": "y_birth",
                             "att_7": "prefer_straight",
                             "politicalid_7": "liberal",
                             "religionid": "religious",
                             }, inplace=True)


    ## give names to categorical values
    df_cat = df_cat.astype('object')
    df_cat['gn_id'].replace({'[1]':'M', '[2]':'F', '[3]': 'Trans_M',
                             '[4]': 'Trans_F', '[5]': 'queer', '[6]': 'other' },
                            inplace=True)

    def CleanGender1(x):
        if x in ['M', 'F']:
            return 'binary'
        else:
            return 'non_binary'
    df_cat['gn_id'] = list(map(CleanGender1, df_cat['gn_id']))

    def CleanGender2(x):
        if x in [1]:
            return 1
        else:
            return 0
    df_cat['straight'] = list(map(CleanGender2, df_cat['straight']))

    def CleanRace(col):
        col = col.replace({1:'American', 2:'East Asian', 3: 'South Asian', 4: 'Pacific',
                           5: 'Black', 6: 'White', 7: 'Other', 8: 'Multiracial'})
        return col
    df_cat['race'] = CleanRace(df_cat['race'])

    df_dummies = ['straight_first', 'fam_gay', 'friend_gay', 'friendly', 'met_gay']

    for col, series in df_cat.iteritems():
        if col in df_dummies:
            series.replace({1:0, 2:1}, inplace=True)

    def CleanSex(x):
        x = x.replace({1: 'F', 2: 'M'})
        return x
    df_cat['sex'] = CleanSex(df_cat['sex'])

    ## get dummies of cat vars
    df_cat = pd.get_dummies(df_cat, drop_first=True)

    df_final = df_num.merge(df_cat, left_index=True, right_index=True)
    df_final.drop(['met_gay'], axis = 1, inplace = True)


    ## FURTHER CLEANING
    df_clean = df_final.copy(deep = True)

    ## 1) outliers
    def removeOutliers(df, col, low, high):
        p1 = low
        p2 = 1-high
        outliers = col.between(col.quantile(p1), col.quantile(p2))
        df = df.loc[outliers]
        return df

    df_clean = removeOutliers(df_clean, df_clean["iat"], 0.025, 0.025)

    ## 2) Change variables
    df_clean["gender_preg_log"] = np.log(df_clean["gender_preg"])

    ## 3) Multicollinearity - remove variables
    # to do
    df_clean.drop(['prefer_straight', 'friend_gay'], axis = 1, inplace = True)

    ### RETURN FINAL DATASETS
    return df_final, df_clean


### 2) MODELING
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

## split and scale |the data
def MODELING(df):

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

    def LinearRegression(y,x):
        x = sm.add_constant(x)
        ols = sm.OLS(y, x).fit()
        ols_table = ols.summary()
        return ols_table

    def MSE_model(y_tr, x_tr_z):
        ols = sm.OLS(y_tr, x_tr_z).fit()
        y_pred_tr = ols.predict(x_tr_z)
        y_tr = y_tr.squeeze()
        mse = ((y_tr - y_pred_tr)**2).sum()/len(y_tr)
        return mse

    def R2_model(y_tr, y_ts, x_tr_z, x_ts_z):
        ols = sm.OLS(y_tr, x_tr_z).fit()
        y_pred_ts = ols.predict(x_ts_z)
        y_ts = y_ts.squeeze()
        r2 = 1 - ((y_ts - y_pred_ts)**2).sum() / ((y_ts - y_ts.mean())**2).sum()
        return r2

    def TableResults2(result):
        table = pd.DataFrame(result.tables[0].data)
        table = table.iloc[:,2:4].T
        table.columns = table.iloc[0]
        table2 = table.iloc[1:,]
        return table2

    ## prepare x and y TRAIN THE MODEL AND SEE STATISTICS
    y = df['iat']
    x = df.drop(['iat'], axis = 1)

    y_tr, y_ts, x_tr_z, x_ts_z = SplitScaleData(y, x, .30, 8)

    ## TEST ON TEST SAMPLE
    ols = sm.OLS(y_tr, x_tr_z).fit()
    ols_table = LinearRegression(y_tr, x_tr_z)
    y_pred_tr = ols.predict(x_tr_z)
    y_pred_ts = ols.predict(x_ts_z)

    ## GET RESULTS
    mean_tr, mean_ts = y_pred_tr.mean(), y_pred_ts.mean()
    std_tr, std_ts = y_pred_tr.std(), y_pred_ts.std()
    diff = y_pred_ts - y_ts.squeeze()
    r2_test = float(TableResults2(LinearRegression(y, x)).iloc[:,1])
    r2_train = R2_model(y_tr, y_ts, x_tr_z, x_ts_z)
    mse = MSE_model(y_pred_ts, y_ts)

    results = {"Mean Train": mean_tr, "Mean test": mean_ts, "Std train": std_tr, "Std test": std_ts,
               "R2 test": r2_test, "R2 train": r2_train, "MSE": mse}

    results = pd.DataFrame.from_dict(results, orient='index', columns=['values'])

    ## PREDICTE VALUES
    y_pred_ts.name = 'y_pred '
    df_pred = pd.merge(y_ts, y_pred_ts, right_index=True, left_index=True)
    df_pred.rename(columns={'iat': 'y_ts'}, inplace=True)

    return results, diff, ols_table, df_pred


## 3) PLOT FUNCTIONS
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

## coefficients and confidence intervals
def PlotOls(table):
    ## prepare the data
    n_coeff = len(table.index)
    table = table.iloc[0:n_coeff, :]
    lower = table['low_ci']
    upper = table['high_ci']
    table['variables'] = table.index.values
    n = len(table)
    l = range(0, n)
    labels = [0] * n
    names = table['variables']
    ##plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot((lower,upper), (l, l), 'ro-', color='blue', linewidth=2, markersize=2)
    plt.scatter(x=table.coef, y=range(0,len(table)))
    ax.tick_params(axis='both', which='both')
    plt.yticks(range(0, n), names)
    plt.plot(labels, l, '--', color='red', linewidth=2, markersize=2)
    ax.set_ylabel('Coefficients', fontsize=10)
    plt.scatter(lower, l, marker='|', color='blue')
    plt.scatter(upper, l, marker='|', color='blue')
    ax.set_facecolor('white')
    plt.title('Coefficients and 95% Confidence Intervals', fontsize=15)
    plt.savefig("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/coefplot.jpeg",
                bbox_inches='tight')


#%% CLEAN DATA
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
pio.renderers.default = "browser"

df = pd.read_stata('iat_sex2020_clean.dta', convert_categoricals=False)
df_raw, df_clean = CLEANING_DATA(df)


#%% target variable distrib
group_labels = ['iat']
hist_target = [df_raw['iat'].to_numpy()]

fig = ff.create_distplot(hist_target, ['IAT score'], bin_size=.05)
fig.show()
fig.update_layout(height=1000, width=1200, title_text="IAT DISTRIBUTION")
fig.write_image("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/iat_distrib.jpeg")
df_raw['iat'].describe().to_csv('C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/descr1.csv',
                                index = True, encoding='utf-8')


#%% CORR MATRIX
corr_matrix = abs(df_raw.corr())

for i in range(0, 22):
    corr_matrix.iloc[i, i:] = 0

fig = px.imshow(corr_matrix, text_auto=False, color_continuous_scale='BuPu')
fig.update_layout(height=600, width=600, title_text="CORRELATION MATRIX")
fig.show()
fig.write_image("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/heatmap.jpeg")


#%% GET RESULTS
results, diff, ols_table, df_pred = MODELING(df_raw)
print(results)


#%% ols plot (coeff + conf. intervals)
ols_table = TableResults(ols_table)
PlotOls(ols_table)


#%% ols plot - bar
bar_table = ols_table[['coef', 'p_val']]
bar_table['sign'] = bar_table['coef'] < 0
bar_table['p_val'] = [0 if i<0.05 else 1 for i in bar_table['p_val']]
bar_table['coef'] = abs(bar_table['coef'])
bar_table.reset_index(inplace=True)
bar_table.sort_values(by=['coef'], ascending = False, inplace=True)
bar_table['ind_var'] = bar_table['ind_var'].astype(str)
fig = px.bar(bar_table, x=bar_table['ind_var'], y=bar_table['coef'], color=bar_table['sign'])
fig.update_layout(height=800, width=1000, title_text="COEFFICIENTS")
fig.write_image("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/ols_coeff.jpeg")
fig.show()
bar_table

#%% statistics
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

plot = results.reset_index(drop=False)
plot.to_csv('C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/ols_results.csv',
            index = True, encoding='utf-8')
plot['color'] = [0,0,1,1,2,2,3]
fig = px.bar(plot, x=plot['index'], y=plot['values'], color=plot['color'])
fig.update_layout(height=800, width=1000, title_text="OLS STATISTICS")
fig.write_image("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/stats_test.jpeg")
fig.show()


#%%  Show the difference between y predicted and y test
import plotly.express as px
fig = px.scatter(x=df_pred['y_ts'], y=df_pred.iloc[:,1], color=df_pred.iloc[:,1])
fig.update_layout(height=1000, width=1000, title_text="Mean and Std. of: y test vs y pred")
fig.update_traces(marker_size=3)
fig.write_image("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/diff.jpeg")
fig.show()


#%% gender prej. distrib
group_labels = ['gender_preg']
hist_target = [df_raw['gender_preg'].to_numpy()]

fig = ff.create_distplot(hist_target, ['gender_preg'], bin_size=.05)
fig.show()
fig.update_layout(height=1000, width=1200, title_text="SEXUAL PREJUDICE")
fig.write_image("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/MidTermProject/results/gender_prej_distrib.jpeg")

#%%
df_raw['friend_gay'].value_counts()
df_raw['gn_id_non_binary'].value_counts()

fig = px.bar(bar_table, x=df_raw['gn_id_non_binary'].value_counts().index, y=df_raw['gn_id_non_binary'].value_counts())
fig = px.bar(bar_table, x=df_raw['friend_gay'].value_counts().index, y=df_raw['friend_gay'].value_counts())
fig.show()
fig.update_layout(height=1000, width=1200, title_text="SEXUAL PREJUDICE")


#%%

