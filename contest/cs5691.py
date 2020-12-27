# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb

rate = pd.read_csv('ratings.csv', parse_dates=['Date'])
rem = pd.read_csv('remarks.csv', parse_dates=['remarkDate'])
rem_sup = pd.read_csv('remarks_supp_opp.csv')
train = pd.read_csv('train.csv', parse_dates=['lastratingdate'])
test = pd.read_csv('test.csv', parse_dates=['lastratingdate'])


def extract_dates(df, column):
    df[column + '_yy'] = df[column].apply(lambda x: x.year)
    df[column + '_mm'] = df[column].apply(lambda x: x.month)
    df[column + '_dd'] = df[column].apply(lambda x: x.day)
    df = df.drop(columns=column)
    return df


def score_function(y_pred, y):
    y_pred = np.asarray(y_pred)
    y = np.asarray(y)
    assert y.shape[0] == y_pred.shape[0]
    n = y.shape[0]
    total = 4 * np.sum(y) + n
    score = 0
    for i in range(n):
        if y_pred[i] == y[i]:
            if y[i] == 1:
                score += 5
            elif y[i] == 0:
                score += 1
    return score / total, score, total


rate = rate.loc[rate['emp'] >= 0]
rem = rem.loc[rem['emp'] >= 0]
rem_sup = rem_sup.loc[rem_sup['emp'] >= 0]

rem = rem.dropna()
rem_sup = rem_sup.dropna()
rate = rate.dropna()

rem = rem.drop_duplicates()
rem_sup = rem_sup.drop_duplicates()
rate = rate.drop_duplicates()

rate = extract_dates(rate, 'Date')
rem = extract_dates(rem, 'remarkDate')
train = extract_dates(train, 'lastratingdate')
test = extract_dates(test, 'lastratingdate')

rem_sup = rem_sup.drop(columns='oppose')

rem['txt_len'] = rem['txt'].apply(lambda x: len(x))
rem = rem.drop(columns='txt')

rate_list = []
emps = rate['emp'].unique()
comps = rate['comp'].unique()
for comp in tqdm(comps):
    comp_wise_list = rate.loc[rate['comp'] == comp]
    for emp in emps:
        r = comp_wise_list.loc[rate['emp'] == emp]
        if r.shape[0]:
            rating = list(r['rating'])
            list_of_rate = np.zeros(4, dtype=np.int32)
            count_dict = Counter(rating)
            for i in range(1, 5):
                list_of_rate[i - 1] = count_dict[i]
            rate_list.append(
                [emp, comp, list_of_rate, r['rating'].shape[0], r.iloc[-1]['Date_yy'], r.iloc[-1]['Date_mm'],
                 r.iloc[-1]['Date_dd']])

rate_df = pd.DataFrame(rate_list, columns=['emp', 'comp', 'rating', 'freq', 'lastratingdate_yy', 'lastratingdate_mm',
                                           'lastratingdate_dd'])

train_rate_list = []
train_rate_freq = []

for _, row in train.iterrows():
    r = rate_df[(rate_df['emp'] == row['emp']) & (rate_df['comp'] == row['comp'])]
    train_rate_list.extend(list(r['rating']))
    train_rate_freq.extend(list(r['freq']))

train['ratings'] = train_rate_list
train['freq'] = train_rate_freq
rate_array = np.asarray(list(train['ratings']))
for i in range(rate_array.shape[1]):
    train['rate_' + str(i + 1)] = rate_array[:, i]
train = train.drop(columns='ratings')

# People who left office have rated lower on average (1-2) is more than the people who stayed. They have rated more (
# 3-4). The median and mean of frequency of people who didn't leave the office is more than the people who left the
# office.

test_rate_list = []
test_rate_freq = []
for _, row in test.iterrows():
    r = rate_df[(rate_df['emp'] == row['emp']) & (rate_df['comp'] == row['comp'])]
    test_rate_list.extend(list(r['rating']))
    test_rate_freq.extend(list(r['freq']))

test['ratings'] = test_rate_list
test['freq'] = test_rate_freq
rate_array = np.asarray(list(test['ratings']))
for i in range(rate_array.shape[1]):
    test['rate_' + str(i + 1)] = rate_array[:, i]
test = test.drop(columns='ratings')

# Ignoring Remark dates for the time being.

remark_len = []
remark_freq = []
support_list = []
total_list = []
for _, row in tqdm(train.iterrows()):
    r = rem[(rem['emp'] == row['emp']) & (rem['comp'] == row['comp'])]
    if r.shape[0]:
        remark_len.append(np.mean(np.asarray(list(r['txt_len']))))
        remarks = r.remarkId.unique()
        support, total = 0, 0
        for remark in remarks:
            rel_list = rem_sup[rem_sup['remarkId'] == remark]
            support += Counter(rel_list['support'])[True]
            total += rel_list.shape[0]
        total_list.append(total)
        support_list.append(support)
    else:
        remark_len.append(0)
        total_list.append(0)
        support_list.append(0)
    remark_freq.append(r.shape[0])

train['max_rem_len'] = remark_len
train['rem_freq'] = remark_freq
train['sup'] = support_list
train['total'] = total_list

remark_len = []
remark_freq = []
support_list = []
total_list = []
for _, row in tqdm(test.iterrows()):
    r = rem[(rem['emp'] == row['emp']) & (rem['comp'] == row['comp'])]
    if r.shape[0]:
        remark_len.append(np.mean(np.asarray(list(r['txt_len']))))
        remarks = r.remarkId.unique()
        support, total = 0, 0
        for remark in remarks:
            rel_list = rem_sup[rem_sup['remarkId'] == remark]
            support += Counter(rel_list['support'])[True]
            total += rel_list.shape[0]
        total_list.append(total)
        support_list.append(support)
    else:
        remark_len.append(0)
        total_list.append(0)
        support_list.append(0)
    remark_freq.append(r.shape[0])

# Now, we are using maximum remark length, if we want we can use, average or some other order statistics.

test['max_rem_len'] = remark_len
test['rem_freq'] = remark_freq
test['sup'] = support_list
test['total'] = total_list

y = train['left']
train_temp = train.drop(columns='left')

mergedata = train_temp.append(test)
testcount = len(test)
count = len(mergedata) - testcount
train_cat = mergedata.copy()

train_cat_one_hot = pd.get_dummies(train_cat, columns=["comp"])

X = train_cat_one_hot[:count]
X_test = train_cat_one_hot[count:]

## Training the Classifier.

iters = 50
y_pred_list = np.zeros((X_test.shape[0], iters))
for i in tqdm(range(iters)):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred_list[:, i] = clf.predict(X_test)

y_pred = np.asarray(np.sum(y_pred_list, axis=1) > 0, dtype=np.int32)

final_preds = pd.DataFrame({'id': list(test['id']), 'left': y_pred})

final_preds.to_csv('preds.csv', index=False)
