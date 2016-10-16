# rmuraglia_statefarm_code.py

"""
Ryan Muraglia's code submission for State Farm data scientist work assignment
Testing period: Oct 13 2016 - Oct 18 2016
"""

# Python version 2.7.10
import numpy as np # version 1.11.1
import pandas as pd # version 0.18.1
import matplotlib.pyplot as plt # version 1.3.1
from sklearn import preprocessing as prep # version 0.19.dev0
from sklearn import linear_model as lm 
from sklearn import tree
# from sklearn import model_selection # if wanted to use GridSearchCV to optimize params
# from sklearn import feature_selection as fselect # if wanted to use f_regression for feature selection
from sklearn.metrics import mean_squared_error
from datetime import datetime
from datetime import timedelta

data_dir = '~/Desktop/sftest/'

"""
Part 1: Clean and prepare data
"""

##
# 1.1: Remove questionable features and samples
##

# load raw data
all_raw = pd.read_csv(data_dir + 'Data for Cleaning & Modeling.csv')

# create a copy of the raw data to clean -- leave raw data untouched
all_working = all_raw.copy() 

# drop variables that I do not want to include in analysis. These include non-informative variables like ID, borrower-write in fields with too many unique difficult to parse responses, and redundant fields like loan grade/loan subgrade or zip code/state
all_working.drop(all_working.columns[[1, 2, 7, 9, 15, 17, 18]], axis=1, inplace=True)

# drop samples that are missing interest rate (won't help for training)
# technically could impute, but risk overfitting if model based, and can generate poor data if non-model based
all_working = all_working[all_working['X1'].notnull()]

# from earlier inspection, we noticed that sample 364111 was missing essentially all the information. drop it.
all_working.drop(364111, inplace=True)

# # many feature's only missing value was for 364111. Let's check which ones still have missing values now
# for col in all_working :
#     print col
#     print np.where(all_working[col].isnull())
#     print len(np.where(all_working[col].isnull())[0])
# # X9 (loan grade) missing 51866 values
# # X12 (home ownership status) missing 51959 values
# # X13 (income) missing 51751 values
# # X25 and X26 (months since negative credit event) missing 185 000 + values
# # X30 (credit utilization) missing 224 values

# remove X25 and X26, as more than half of the data is missing (and is not simply due to missing values indicating they never had a negative mark)
# remove X12 as it didn't appear to be predictive
# keep X13 and try to impute
all_working.drop(all_working.columns[[7, 17, 18]], axis=1, inplace=True)

# also drop X23: in test set, dates are encoded differently, and appear to be missing year information, which is critical
all_working.drop('X23', axis=1, inplace=True)

# also drop X15: in test set, dates are encoded differently, and it is unclear if the 15 refers to a date or a year.
# given that X15 is actually a reasonably strong predictor, I would rather not give potentially erroneous information to the model
# all_working.drop('X15', axis=1, inplace=True)

##
# 1.2: Parse data to be suitable for computation
##

def strip_prc(series) :
    return series.str.strip('%').astype('float')

def strip_dollar(series) :
    return series.str.replace('[\$,]', '').astype('float')

def encode_ordinal(series) :
    # also use this for binary
    le = prep.LabelEncoder()
    le.fit(series)
    ordinal_labels = le.transform(series)
    series_out = pd.Series(ordinal_labels, index=series.index, name=series.name)
    return le, series_out

def encode_categorical(series) :
    le = prep.LabelEncoder()
    le.fit(series)
    series_ints = le.transform(series)
    enc = prep.OneHotEncoder(sparse=False)
    enc.fit(series_ints.reshape(-1,1))
    series_mat = enc.transform(series_ints.reshape(-1,1))
    colnames = [series.name + '-' + str(i) for i in xrange(len(le.classes_))]
    series_df = pd.DataFrame(series_mat, index=series.index, columns=colnames)
    return le, enc, series_df

def parse_X11(series) :
    # requires different parser from ordinal to handle inappropriate sort order and differently marked missing values
    # first replace n/a values with samples drawn from the empirical distribution
    counts = series.value_counts()
    counts.drop('n/a', inplace=True)
    fracs = counts/sum(counts)
    series_out = pd.Series(np.empty(len(series)), index=series.index, name=series.name)
    for i in series.index :
        if series[i] == 'n/a' :
            x = np.random.choice(fracs.index, size=1, p=fracs.values)[0]
        else : 
            x = series[i]
        if '<' in x : 
            series_out[i] = 0
        elif '+' in x :
            series_out[i] = 11
        else :
            series_out[i] = x.split()[0]
    return fracs, series_out

def date_to_ordinal(series) :
    parsed_dates = [datetime.strptime(i, '%b-%y') for i in series]
    century = timedelta(days = 36525)
    corrected_dates = [i-century if i>datetime.today() else i for i in parsed_dates]
    ordinals = [i.toordinal() for i in corrected_dates]
    series_out = pd.Series(ordinals, index=series.index, name=series.name)
    return series_out

X1_clean = strip_prc(all_working['X1'])
X4_clean = strip_dollar(all_working['X4'])
X5_clean = strip_dollar(all_working['X5'])
# X5_clean = (X4_clean>X5_strip).astype('int') # if encode as bool, change previous line assingment to X5_strip
# X5_clean.rename('X5', inplace=True)
X6_clean = strip_dollar(all_working['X6'])
# X6_clean = (X5_strip>X6_strip).astype('int') # same as for X5
# X6_clean.rename('X6', inplace=True)
X7_le, X7_clean = encode_ordinal(all_working['X7'])
X9_le, X9_clean = encode_ordinal(all_working['X9']) # na encoded as 0 - see X9_le.classes_
X11_fracs, X11_clean = parse_X11(all_working['X11'])
X13_clean = all_working['X13'].copy()
X14_le, X14_enc, X14_clean = encode_categorical(all_working['X14'])
X15_clean = date_to_ordinal(all_working['X15'])
X17_le, X17_enc, X17_clean = encode_categorical(all_working['X17'])
X20_le, X20_enc, X20_clean = encode_categorical(all_working['X20'])
X21_clean = all_working['X21'].copy()
X22_clean = all_working['X22'].copy()
# X23_alt = X15_clean - date_to_ordinal(all_working['X23']) # for X23, do age of earliest credit line at date of loan issuance
# X23_alt.rename('X23-alt', inplace=True)
X24_clean = all_working['X24'].copy()
X27_clean = all_working['X27'].copy()
X28_clean = all_working['X28'].copy()
X29_clean = all_working['X29'].copy()
X30_clean = strip_prc(all_working['X30'])
X31_clean = all_working['X31'].copy()
X32_le, X32_clean = encode_ordinal(all_working['X32'])

all_clean = pd.concat([X1_clean, X4_clean, X5_clean, X6_clean, X7_clean, X9_clean, X11_clean, X13_clean, X14_clean, X15_clean, X17_clean, X20_clean, X21_clean, X22_clean, X24_clean, X27_clean, X28_clean, X29_clean, X30_clean, X31_clean, X32_clean], axis=1) 

##
# 1.3: Impute missing values
##

# for X9 (loan grade), missing values are currently encoded as 0. 
# for each 0 assign label for nearest median
X9_medians = all_clean.groupby(['X9'])['X1'].median()[1:]

def impute_X9(series9, series1, median_vec=X9_medians) :
    s_copy = series9.copy()
    na_inds = np.where(series9==0)[0]
    for i in na_inds :
        s_copy.iloc[i] = np.argmin(abs(series1.iloc[i] - median_vec))
    return s_copy

all_clean['X9'] = impute_X9(all_clean['X9'], all_clean['X1'])

# for X13 (income), impute as median. Consider doing gaussian centered on median to add noise
income_med = all_clean['X13'].median() 
all_clean['X13'].fillna(income_med, inplace=True) 

# for X30 (utilization rate), just impute with median. 
X30_median = np.nanmedian(all_clean['X30'])
na_inds_30 = np.where(all_clean['X30'].isnull())[0]
all_clean['X30'].iloc[na_inds_30] = X30_median

##
# 1.4: Final preparations
##

# standardize features
y_clean = all_clean.iloc[:, 0]
X_clean = all_clean.iloc[:, 1:]

scaler = prep.StandardScaler()
scaler.fit(X_clean)
X_std = scaler.transform(X_clean)
X_std = pd.DataFrame(X_std, index=X_clean.index, columns=X_clean.columns)

# split into training and cross-validation sets
X_train = X_std.sample(frac=0.75)
X_xval = X_std.drop(X_train.index)
y_train = y_clean[X_train.index]
y_xval = y_clean.drop(X_train.index)

"""
Part 2: Build LASSO regression model
The LASSO is a linear model that enforces shrinkage and sparsity in the coefficients. Because of this, it is a very convenient model, as it accomplishes feature selection and regression simultaneously.
"""

# the Lasso has one parameter, alpha, which controls the level of shrinkage. We will choose the best value for alpha based on minimum MSE on a cross validation set

num_alphas = 15
alphas = np.logspace(-4, 0, num_alphas)
lasso_mse = np.empty(num_alphas)

# fit a LASSO for each alpha value
for i in xrange(num_alphas) :
    lasso = lm.Lasso(alpha = alphas[i])
    lasso.fit(X_train, y_train)
    lasso_mse[i] = mean_squared_error(y_xval, lasso.predict(X_xval))
    # print lasso_mse

# choose alpha that minimizes MSE and fit model with that alpha to full test set
best_alpha = alphas[np.argmin(lasso_mse)]
print "The alpha selected for LASSO was %f with a crossvalidation MSE of %f" % (best_alpha, min(lasso_mse))
lasso = lm.Lasso(alpha = best_alpha)
lasso.fit(X_std, y_clean)

"""
Part 3: Build decision tree regression model
Decision trees can also be used for regression, and not just classification. Decision trees are convenient as they are intuitively easy to understand, and they perform very well with categorical data.
"""

# an important parameter for decision trees is their maximum depth, which influences the complexity of the model. We will use a similar crossvalidation MSE minimization routine to select a model at the sweet spot of complexity.

max_depth = 20
depths = np.arange(max_depth) + 1
dtree_mse = np.empty(max_depth)

# fit a tree for each depth
for i in xrange(max_depth) :
    dtree = tree.DecisionTreeRegressor(max_depth = depths[i])
    dtree.fit(X_train, y_train)
    dtree_mse[i] = mean_squared_error(y_xval, dtree.predict(X_xval))
    # print dtree_mse

# choose depth that minimizes MSE and fit model with that depth to full test set
best_depth = depths[np.argmin(dtree_mse)]
print "The max-depth selected for the decision tree was %i with a crossvalidation MSE of %f" % (best_depth, min(dtree_mse))
dtree = tree.DecisionTreeRegressor(max_depth = best_depth)
dtree.fit(X_std, y_clean)

"""
Part 4: load and prepare test data
"""

# load test data
test_raw = pd.read_csv(data_dir + 'Holdout for Testing.csv')

# process data in same manner as training data
test_working = test_raw.copy()
test_working.drop(test_working.columns[[1, 2, 7, 9, 15, 17, 18]], axis=1, inplace=True)
test_working.drop(test_working.columns[[7, 17, 18]], axis=1, inplace=True)
test_working.drop('X23', axis=1, inplace=True)
# test_working.drop('X15', axis=1, inplace=True)

def encode_test_cat(series, le, enc) :
    series_ints = le.transform(series)
    series_mat = enc.transform(series_ints.reshape(-1,1))
    colnames = [series.name + '-' + str(i) for i in xrange(len(le.classes_))]
    series_df = pd.DataFrame(series_mat, index=series.index, columns=colnames)
    return series_df

def encode_test_ord(series, le) :
    series_out = pd.Series(le.transform(series), index=series.index, name=series.name)
    return series_out

def encode_test_11(series, fracs) :
    series_out = pd.Series(np.empty(len(series)), index=series.index, name=series.name)
    for i in series.index :
        if series[i] == 'n/a' :
            x = np.random.choice(fracs.index, size=1, p=fracs.values)[0]
        else : 
            x = series[i]
        if '<' in x : 
            series_out[i] = 0
        elif '+' in x :
            series_out[i] = 11
        else :
            series_out[i] = x.split()[0]
    return series_out

def date_to_ordinal2(series) :
    parsed_dates = [datetime.strptime(i, '%y-%b') for i in series]
    century = timedelta(days = 36525)
    corrected_dates = [i-century if i>datetime.today() else i for i in parsed_dates]
    ordinals = [i.toordinal() for i in corrected_dates]
    series_out = pd.Series(ordinals, index=series.index, name=series.name)
    return series_out

X4_test = strip_dollar(test_working['X4'])
X5_test = strip_dollar(test_working['X5']) 
# X5_test = (X4_test>X5_proc).astype('int') # if encode as bool, change previous line assingment to X5_proc
# X5_test.rename('X5', inplace=True)
X6_test = strip_dollar(test_working['X6'])
# X6_test = (X5_test>X6_proc).astype('int') # same as X5 above
# X6_test.rename('X6', inplace=True)
X7_test = encode_test_ord(test_working['X7'], X7_le)
X9_test = encode_test_ord(test_working['X9'], X9_le)
X11_test = encode_test_11(test_working['X11'], X11_fracs)
X13_test = test_working['X13'].copy()
X14_test = encode_test_cat(test_working['X14'], X14_le, X14_enc)
X15_test = date_to_ordinal2(test_working['X15'])
X17_test = encode_test_cat(test_working['X17'], X17_le, X17_enc)
X20_test = encode_test_cat(test_working['X20'], X20_le, X20_enc)
X21_test = test_working['X21'].copy()
X22_test = test_working['X22'].copy()
X24_test = test_working['X24'].copy()
X27_test = test_working['X27'].copy()
X28_test = test_working['X28'].copy()
X29_test = test_working['X29'].copy()
X30_test = strip_prc(test_working['X30'])
X31_test = test_working['X31'].copy()
X32_test = encode_test_ord(test_working['X32'], X32_le)

test_clean = pd.concat([X4_test, X5_test, X6_test, X7_test, X9_test, X11_test, X13_test, X14_test, X15_test, X17_test, X20_test, X21_test, X22_test, X24_test, X27_test, X28_test, X29_test, X30_test, X31_test, X32_test], axis=1)

# impute missing utilization rates
test_clean['X30'].fillna(X30_median, inplace=True)

# standardize
test_std = scaler.transform(test_clean)
X_test = pd.DataFrame(test_std, index=test_clean.index, columns=test_clean.columns)

"""
Part 5: make predictions on unseen data
"""

lasso_predictions = lasso.predict(X_test)
dtree_predictions = dtree.predict(X_test)

predictions_out = pd.DataFrame({'LASSO' : lasso_predictions, 'DTree' : dtree_predictions})
predictions_out.to_csv('Results from Ryan Muraglia.csv', index=False)

"""
Part 6: optional diagnostic looks at models
"""

# print out most features that explain the most variance
lasso_coefs = np.column_stack((X_train.columns, lasso.coef_))
lasso_top = abs(lasso_coefs[:,1]).argsort()[::-1][0:5] # reverse sort order and get top 5
print 'The most predictive LASSO variables are: '
print lasso_coefs[lasso_top, :]

dtree_importance = np.column_stack((X_train.columns, dtree.feature_importances_))
dtree_top = dtree_importance[:,1].argsort()[::-1][0:5]
print 'The most important decision tree variables are: '
print dtree_importance[dtree_top, :]

# test against simple linear model using only loan grade
loan_grade_train = X_train['X9'].reshape(-1,1)
loan_grade_xval = X_xval['X9'].reshape(-1,1)
loan_grade_test = X_test['X9'].reshape(-1,1)
ols_lg = lm.LinearRegression()
ols_lg.fit(loan_grade_train, y_train)
print 'An ordinary least squares linear regression using only loan subgrade (X9) as a predictor has a crossvalidation MSE of %f' % (mean_squared_error(y_xval, ols_lg.predict(loan_grade_xval)))
loan_grade_predict = ols_lg.predict(loan_grade_test)

# visualize agreement between models
fig = plt.figure()
ax1 = plt.subplot(131)
ax1.plot(loan_grade_predict, lasso_predictions, marker='o', linestyle='none')
ax1.set_xlabel('Simple OLS'); ax1.set_ylabel('LASSO')
ax2 = plt.subplot(132)
ax2.plot(loan_grade_predict, dtree_predictions, marker='o', linestyle='none')
ax2.set_xlabel('Simple OLS'); ax2.set_ylabel('DTree')
ax3 = plt.subplot(133)
ax3.plot(lasso_predictions, dtree_predictions, marker='o', linestyle='none')
ax3.set_xlabel('LASSO'); ax3.set_ylabel('DTree')
plt.show()

# look at decision tree structure and decision rules for a shallow tree 
import pydotplus
from sklearn.externals.six import StringIO  

mini_tree = tree.DecisionTreeRegressor(max_depth=4)
mini_tree.fit(X_train, y_train)

dot_data = StringIO()
tree.export_graphviz(mini_tree, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("minitree.pdf") 

