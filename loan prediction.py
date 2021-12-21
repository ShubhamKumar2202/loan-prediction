import pandas as pd
import numpy as np  # For mathematical calculations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs
import warnings  # To ignore any warnings

warnings.filterwarnings("ignore")

# Train file will be used for training the model, i.e. our model will learn from this file. It contains all the
# independent variables and the target variable. Test file contains all the independent variables, but not the target
# variable. We will apply the model to predict the target variable for the test data. Sample submission file contains
# the format in which we have to submit our predictions. Reading data

train = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
test = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

# Let’s make a copy of train and test data so that even if we
# have to make any changes in these datasets we would not lose the original datasets.
train_original = train.copy()
test_original = test.copy()

train.keys()
print(train.columns)
print(test.columns)

# print data_types for each variables
print(train.dtypes)
print(train.shape, test.shape)

# Analysis
train['Loan_Status'].value_counts()

# Normalise can be set to true to print the proportions instead of Numbers.
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()

# visualize each variable separately
# Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)
# Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)
# Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount,
# Loan_Amount_Term)
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Credit_History')
plt.show()

# let’s visualize the ordinal variables.

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24, 6), title='Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(figsize=(24, 6), title='Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(24, 6), title='Property_Area')
# Lets visualise Numerical data

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16, 5))

# The boxplot confirms the presence of a lot of extreme values.
# This can be attributed to the income disparity in the society.
# this can be driven by the fact that we are looking at people with different education levels.
# Let us segregate them by Education:

train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("")

# Let’s look at the Coapplicant income distribution.

plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16, 5))

# Let’s look at the distribution of LoanAmount variable.

plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16, 5))

# Missing Values and Outliers Treatements

train.info()
train.isnull().sum()

# There are missing values in Gender, Married, Dependents, Self_Employed, LoanAmount,
# Loan_Amount_Term and Credit_History features.
# 1) We will treat the missing values in all the features one by one.
# 2) We can consider these methods to fill the missing values:

# a)For numerical variables: imputation using mean or median
# b)For categorical variables: imputation using mode

# There are very less missing values in Gender, Married, Dependents, Credit_History and Self_Employed features so,
# we can fill them using the mode of the features.

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train.isnull().sum()

# let’s try to find a way to fill the missing values in Loan_Amount_Term.
# We will look at the value count of the Loan amount term variable.

train['Loan_Amount_Term'].value_counts()
# It can be seen that in loan amount term variable, the value of 360 is repeating the most. So we will replace the
# missing values in this variable using the mode of this variable.
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train.isnull().sum()

# As we can see that all the missing values have been filled in the train dataset. Let’s fill all the missing values in
# the test dataset too with the same approach.

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# Let’s visualize the effect of log transformation. We will do the similar changes to the test file simultaneously.

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

# Bivariate Analysis
# Categorical Independent Variable vs Target Variable
# First of all we will find the relation between target variable and categorical independent variables.
# Let us look at the stacked bar plot now which will give us the proportion of approved and unapproved loans.

train.describe()
train.shape

train.dropna()
train.shape

Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))

# It can be inferred that the proportion of male and female applicants is more or less same
# for both approved and unapproved loans.
# Now let us visualize the remaining categorical variables vs target variable.

Married = pd.crosstab(train['Married'], train['Loan_Status'])
Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
Education = pd.crosstab(train['Education'], train['Loan_Status'])
Self_Employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar', stacked='True', figsize=(4, 4))

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind='bar', stacked='True', figsize=(4, 4))

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))

# Lets look at the relationship between remaining categorical independent variables and Loan_Status.

Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

#  let’s visualize numerical independent variables with respect to target variable.
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

bins = [0, 2500, 4000, 6000, 81000]
group = ['HIgh', 'Average', 'Low', 'Very high']
train['Income_bin'] = pd.cut(train['ApplicantIncome'], bins, right=True, labels=group)

Income_bin = pd.crosstab(train['Income_bin'], train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')

Income_bin = pd.crosstab(train['Income_bin'], train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
plt.ylabel('Percentage')

# We will analyze the coapplicant income and loan amount variable in similar manner.

bins = [0, 1000, 3000, 42000]
group = ['Low', 'Average', 'High']
train['Coapplicant_Income_bin'] = pd.cut(train['CoapplicantIncome'], bins, include_lowest=True, labels=group)
Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'], train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome')
P = plt.ylabel('Percentage')

# Let us combine the Applicant Income and Coapplicant Income and see the combined effect of
# Total Income on the Loan_Status.

train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
bins = [0, 2500, 4000, 6000, 81000]
group = ['Very High', 'High', 'Low', 'Average']
train['Total_Income_bin'] = pd.cut(train['Total_Income'], bins, labels=group)
Total_Income_bin = pd.crosstab(train['Total_Income_bin'], train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')

# Let’s visualize the Loan amount variable.

bins = [0, 100, 200, 700]
group = ['Average', 'Low', 'High']
train['LoanAmount_bin'] = pd.cut(train['LoanAmount'], bins, labels=group)

LoanAmount_bin = pd.crosstab(train['LoanAmount_bin'], train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

train = train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'],
                   axis=1)
train['Dependents'].replace('3+', 3, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)
train['Loan_Status'].replace('N', 0, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True)

# lets look at the correlation between all the numerical variables
# The variables with darker color means their correlation is more.

matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");

# Model Building : Part I
# Let us make our model to predict the target variable.
# We will start with Logistic Regression which is used for predicting binary outcome.
# Lets drop the Loan_ID variable as it do not have any effect on the loan status

train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

X = train.drop('Loan_Status', 1)
y = train.Loan_Status

# we will make dummy variables for the categorical variables

X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# We will use the train_test_split function from sklearn to divide our train dataset.


from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

# The dataset has been divided into training and validation part.
# Let us import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)

# Let’s predict the Loan_Status for validation set and calculate its accuracy.
pred_cv = model.predict(x_cv)

# Let us calculate how accurate our predictions are by calculating the accuracy.
accuracy_score(y_cv, pred_cv)

# Let’s make predictions for the test dataset.

pred_test = model.predict(test)
# Lets import the submission file which we have to submit on the solution checker.

submission = pd.read_csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
# We only need the Loan_ID and the corresponding Loan_Status for the final submission.
# we will fill these columns with the Loan_ID of test dataset and the predictions that we made

submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']
# Logistic Regression using stratified k-folds cross validation

from sklearn.model_selection import StratifiedKFold

# Now let’s make a cross validation logistic model with stratified 5 folds and make predictions for test dataset.
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
pred_test = model.predict(test)
pred = model.predict_proba(xvl)[:, 1]

# The mean validation accuracy for this model turns out to be 0.81. Let us visualize the roc curve.
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, label="validation, auc=" + str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)

submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']
# Remember we need predictions in Y and N. So let’s convert 1 and 0 to Y and N.

submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)

# Lets convert the submission to .csv format and make submission to check the accuracy on the leaderboard.

pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('Logistic1.csv')

train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']

# Let’s check the distribution of Total Income.

sns.distplot(train['Total_Income']);
# let’s take the log transformation to make the distribution normal.

train['Total_Income_log'] = np.log(train['Total_Income'])
sns.distplot(train['Total_Income_log']);
test['Total_Income_log'] = np.log(test['Total_Income'])

# Now the distribution looks much closer to normal and effect of extreme values has been significantly subsided.
# Let’s create the EMI feature now.

train['EMI'] = train['LoanAmount'] / train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount'] / test['Loan_Amount_Term']

# Let’s check the distribution of EMI variable.
sns.distplot(train['EMI']);

# Let us create Balance Income feature now and check its distribution.

train['Balance Income'] = train['Total_Income'] - (train['EMI'] * 1000)  # Multiply with 1000 to make the units equal
test['Balance Income'] = test['Total_Income'] - (test['EMI'] * 1000)
sns.distplot(train['Balance Income']);

train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

X = train.drop('Loan_Status', 1)
y = train.Loan_Status

# Logistic Regression

i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
pred_test = model.predict(test)
pred = model.predict_proba(xvl)[:, 1]

submission['Loan_Status'] = pred_test  # filling Loan_Status with predictions
submission['Loan_ID'] = test_original['Loan_ID']  # filling Loan_ID with test Loan_ID
# replacing 0 and 1 with N and Y
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
# Converting submission file to .csv format
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('Log2.csv')

from sklearn import tree

i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = tree.DecisionTreeClassifier(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
pred_test = model.predict(test)

submission['Loan_Status'] = pred_test  # filling Loan_Status with predictions
submission['Loan_ID'] = test_original['Loan_ID']  # filling Loan_ID with test Loan_ID
# replacing 0 and 1 with N and Y
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
# Converting submission file to .csv format
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('Decision Tree.csv')

# Random Forest

from sklearn.ensemble import RandomForestClassifier

i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = RandomForestClassifier(random_state=1, max_depth=10)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
pred_test = model.predict(test)
# We will tune the max_depth and n_estimators parameters. max_depth decides the maximum depth of the tree and
# n_estimators decides the number of trees that will be used in random forest model.

from sklearn.model_selection import GridSearchCV

# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an
# interval of 20 for n_estimators
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=1)
# Fit the grid search model
grid_search.fit(x_train, y_train)

GridSearchCV(cv=None, error_score='raise',
             estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                              max_depth=None, max_features='auto', max_leaf_nodes=None,
                                              min_impurity_decrease=0.0, min_impurity_split=None,
                                              min_samples_leaf=1, min_samples_split=2,
                                              min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                              oob_score=False, random_state=1, verbose=0, warm_start=False),
             fit_params=None, iid=True, n_jobs=1,
             param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                         'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
             scoring=None, verbose=0)
# Estimating the optimized value
grid_search.best_estimator_

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=3, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=41, n_jobs=1,
                       oob_score=False, random_state=1, verbose=0, warm_start=False)
# So, the optimized value for the max_depth variable is 3 and for n_estimator is 41.
# Now let’s build the model using these optimized values.

i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = X.loc[train_index], X.loc[test_index]
    ytr, yvl = y[train_index], y[test_index]

    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    i += 1
pred_test = model.predict(test)
pred2 = model.predict_proba(test)[:, 1]

submission['Loan_Status'] = pred_test  # filling Loan_Status with predictions
submission['Loan_ID'] = test_original['Loan_ID']  # filling Loan_ID with test Loan_ID
# replacing 0 and 1 with N and Y
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
# Converting submission file to .csv format
pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('Random Forest.csv')

# Let us find the feature importance now, i.e. which features are most important for this problem.
# We will use featureimportances attribute of sklearn to do so.

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12, 8))

plt.show()
