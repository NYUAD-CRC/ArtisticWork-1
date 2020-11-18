#importing libraries and the data set:
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.linear_model import LinearRegression , LogisticRegression,BayesianRidge,Perceptron
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

#load the inoput from the file
df = pd.read_csv("./df_for_ml_stata.csv", low_memory=False) #Reading the dataset in a dataframe using Pandas
#df = dff[0:1000]
#Quick Data Exploration


#summary of numerical fieldsprint(df.head(10))
print(df.describe())

#Distribution analysis
#df.boxplot(column='price_usd', by = 'year',figsize=(5,3))
df.boxplot(column='price_usd', by = 'year')
figure = plt.gcf()
#figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
#figure.set_size_inches(80, 60)
figure.set_size_inches(12, 8)
plt.savefig('boxplot of price grouped by year.pdf')
plt.savefig('boxplot of price grouped by year.png', bbox_inches='tight', dpi=100)


#Check missing values in the dataset
df.apply(lambda x: sum(x.isnull()), axis=0)

df.replace(np.nan,0, inplace=True)

my_tab = pd.crosstab(index = df["year"],  # Make a crosstab
                              columns="price_usd")      # Name the count column

my_tab.plot.bar(y='price_usd', rot=0,color = 'g')
plt.title('barplot of price by year')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('barplot of price by year.png', bbox_inches='tight', dpi=100)

#Building a Predictive Model
outcome = ['price_usd']
predictors = df.columns.difference(['price_usd'])


## CORR check ::::
inp1 = df.columns

##Pair wise correlation check
#sns.pairplot(df[inp1])
#plt.show()
corr = df[inp1].corr()
plt.figure(figsize=(40,40))
sns.heatmap(corr, vmin=-1, vmax=1, center=0,xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(20,220, n=500),annot=True, fmt="f")
#plt.show()
figure = plt.gcf()
figure.set_size_inches(80, 60)
plt.savefig('corrplot.png', bbox_inches='tight', dpi=100)

correlation_mat = df[inp1].corr()
corr_pairs = correlation_mat.unstack()
corr_pairs.to_csv('corr_pairs.csv',encoding='utf-8')
print(corr_pairs)
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
sorted_pairs.to_csv('sorted_pairs.csv',encoding='utf-8')
print(sorted_pairs)
negative_pairs = sorted_pairs[sorted_pairs < 0]
negative_pairs.to_csv('negative_pairs.csv',encoding='utf-8')
print(negative_pairs)
strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]
strong_pairs.to_csv('strong_pairs.csv',encoding='utf-8')
print(strong_pairs)
plt.clf()

#setting input and ouput for the model
X1=df[predictors]
y=df[outcome]

# normalising/scaling all the values for optimisation
scaler = StandardScaler()
X = scaler.fit_transform( X1 )

# linear regression feature importance
# define the model
model = LinearRegression(n_jobs=30)
#model = BayesianRidge()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.figure(figsize=(9,6))
plt.bar(x=[int(x) for x in range(len(importance))], height=importance,color = 'c')
plt.xticks(rotation=45)
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('linearReg_FeaImp.png', bbox_inches='tight', dpi=100)
plt.clf()

coeff_reg = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(model.coef_[0])})
coeff_reg.to_csv('coeff_reg.csv',encoding='utf-8',index=False)
plt.figure(figsize=(9,6))
plt.bar(x=coeff_reg['Feature'], height=coeff_reg['Coefficients'],color = 'b')
plt.xticks(rotation=45)
plt.title('Features importance by linear regression')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('linearReg_FeaImptance.png', bbox_inches='tight', dpi=100)
plt.clf()

feat_importances = pd.Series(model.coef_[0], index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh',color = list('rgbkymc'))
plt.title('Top 25 features of  linear regression')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('linearReg_Top25.png', bbox_inches='tight', dpi=100)
plt.clf()

# decision tree for feature importance on a regression problem
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.figure(figsize=(9,6))
plt.bar(x=[int(x) for x in range(len(importance))], height=importance,color = 'c')
plt.xticks(rotation=45)
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('DecisionTreeReg_FeaImp.png', bbox_inches='tight', dpi=100)
plt.clf()

coeff_reg = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(model.feature_importances_)})
coeff_reg.to_csv('coeff_DecisionTreeReg.csv',encoding='utf-8',index=False)
plt.figure(figsize=(9,6))
plt.bar(x=coeff_reg['Feature'], height=coeff_reg['Coefficients'],color = 'b')
plt.xticks(rotation=45)
plt.title('Features importance by DecisionTreeReg')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('DecisionTreeReg_FeaImptance.png', bbox_inches='tight', dpi=100)
plt.clf()

feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh',color = list('rgbkymc'))
plt.title('Top 25 features of DecisionTreeReg')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('DecisionTreeReg_Top25.png', bbox_inches='tight', dpi=100)
plt.clf()

# random forest for feature importance on a regression problem
# define the model
model = RandomForestRegressor(n_jobs=30)
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.figure(figsize=(9,6))
plt.bar(x=[int(x) for x in range(len(importance))], height=importance,color = 'c')
plt.xticks(rotation=45)
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('Randomforest_FeaImp.png', bbox_inches='tight', dpi=100)
plt.clf()

coeff_reg = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(model.feature_importances_)})
coeff_reg.to_csv('coeff_Randomforest.csv',encoding='utf-8',index=False)
plt.figure(figsize=(9,6))
plt.bar(x=coeff_reg['Feature'], height=coeff_reg['Coefficients'],color = 'b')
plt.xticks(rotation=45)
plt.title('Features importance by Randomforest')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('Randomforest_FeaImptance.png', bbox_inches='tight', dpi=100)
plt.clf()

feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh',color = list('rgbkymc'))
plt.title('Top 25 features of Randomforest')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('Randomforest_Top25.png', bbox_inches='tight', dpi=100)
plt.clf()


# xgboost for feature importance on a regression problem
# define the model
model = XGBRegressor(n_jobs=30)
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.figure(figsize=(9,6))
plt.bar(x=[int(x) for x in range(len(importance))], height=importance,color = 'c')
plt.xticks(rotation=45)
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('XGBRegressor_FeaImp.png', bbox_inches='tight', dpi=100)
plt.clf()

coeff_reg = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(model.feature_importances_)})
coeff_reg.to_csv('coeff_XGBRegressor.csv',encoding='utf-8',index=False)
plt.figure(figsize=(9,6))
plt.bar(x=coeff_reg['Feature'], height=coeff_reg['Coefficients'],color = 'b')
plt.xticks(rotation=45)
plt.title('Features importance by XGBRegressor')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('XGBRegressor_FeaImptance.png', bbox_inches='tight', dpi=100)
plt.clf()

feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh',color = list('rgbkymc'))
plt.title('Top 25 features of XGBRegressor')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('XGBRegressor_Top25.png', bbox_inches='tight', dpi=100)
plt.clf()

# GradientBoostingRegressor for feature importance on a regression problem
# define the model
model = GradientBoostingRegressor(random_state=0)
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.figure(figsize=(9,6))
plt.bar(x=[int(x) for x in range(len(importance))], height=importance,color = 'c')
plt.xticks(rotation=45)
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('GradientBoostingRegressor_FeaImp.png', bbox_inches='tight', dpi=100)
plt.clf()

coeff_reg = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(model.feature_importances_)})
coeff_reg.to_csv('coeff_GradientBoostingRegressor.csv',encoding='utf-8',index=False)
plt.figure(figsize=(9,6))
plt.bar(x=coeff_reg['Feature'], height=coeff_reg['Coefficients'],color = 'b')
plt.xticks(rotation=45)
plt.title('Features importance by GradientBoostingRegressor')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('GradientBoostingRegressor_FeaImptance.png', bbox_inches='tight', dpi=100)
plt.clf()

feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh',color = list('rgbkymc'))
plt.title('Top 25 features of GradientBoostingRegressor')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('GradientBoostingRegressor_Top25.png', bbox_inches='tight', dpi=100)
plt.clf()


# Perceptron feature importance
# define the model
model = Perceptron(n_jobs=30)
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.figure(figsize=(9,6))
plt.bar(x=[int(x) for x in range(len(importance))], height=importance,color = 'c')
plt.xticks(rotation=45)
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('Perceptron_FeaImp.png', bbox_inches='tight', dpi=100)
plt.clf()

coeff_reg = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(model.coef_[0])})
coeff_reg.to_csv('coeff_Perceptron.csv',encoding='utf-8',index=False)
plt.figure(figsize=(9,6))
plt.bar(x=coeff_reg['Feature'], height=coeff_reg['Coefficients'],color = 'b')
plt.xticks(rotation=45)
plt.title('Features importance by Perceptron')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('Perceptron_FeaImptance.png', bbox_inches='tight', dpi=100)
plt.clf()

feat_importances = pd.Series(model.coef_[0], index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh',color = list('rgbkymc'))
plt.title('Top 25 features of Perceptron')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('Perceptron_Top25.png', bbox_inches='tight', dpi=100)
plt.clf()


# AdaBoostRegressor feature importance
# define the model
model = AdaBoostRegressor(random_state=0, n_estimators=100)
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.figure(figsize=(9,6))
plt.bar(x=[int(x) for x in range(len(importance))], height=importance,color = 'c')
plt.xticks(rotation=45)
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('AdaBoostRegressor_FeaImp.png', bbox_inches='tight', dpi=100)
plt.clf()

coeff_reg = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(model.feature_importances_)})
coeff_reg.to_csv('coeff_AdaBoostRegressor.csv',encoding='utf-8',index=False)
plt.figure(figsize=(9,6))
plt.bar(x=coeff_reg['Feature'], height=coeff_reg['Coefficients'],color = 'b')
plt.xticks(rotation=45)
plt.title('Features importance by AdaBoostRegressor')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('Perceptron_AdaBoostRegressor.png', bbox_inches='tight', dpi=100)
plt.clf()

feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(25).plot(kind='barh',color = list('rgbkymc'))
plt.title('Top 25 features of AdaBoostRegressor')
figure = plt.gcf()
figure.set_size_inches(12, 8)
plt.savefig('AdaBoostRegressor_Top25.png', bbox_inches='tight', dpi=100)
plt.clf()


'''
# permutation feature importance with knn for regression
# define the model
model = RandomForestRegressor(n_jobs=30)
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)

plt.savefig('permutations.png')
#plt.show()
coeff_per = pd.DataFrame({"Feature":X1.columns,"Coefficients":np.transpose(results.importances_mean)})
coeff_per.to_csv('coeff_per.csv',encoding='utf-8',index=False)
'''

# evaluation of a model using all features
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
print('RandomForestRegressor traning and evaluation')
model = RandomForestRegressor(n_jobs=30)
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
Variance_score = model.score(X_test, y_test)
print('Variance score for full features: %.2f' % Variance_score)

# evaluation of a model using selected features chosen with random forest importance
# feature selection
def select_features(X_train, y_train, X_test,model):
    # configure to select a subset of features
    fs = SelectFromModel(model, max_features=15)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
model = RandomForestRegressor(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
Variance_score = model.score(X_test_fs, y_test)
print('Variance score for top features: %.2f' % Variance_score)

#LinearRegression testing
# fit the model
print('LinearRegression traning and evaluation')
model = LinearRegression(n_jobs=30)
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
Variance_score = model.score(X_test, y_test)
print('Variance score for full features: %.2f' % Variance_score)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
model = LinearRegression(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
Variance_score = model.score(X_test_fs, y_test)
print('Variance score for top features: %.2f' % Variance_score)

#DecisionTreeRegressor() testing
# fit the model
print('DecisionTreeRegressor traning and evaluation')
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
Variance_score = model.score(X_test, y_test)
print('Variance score for full features: %.2f' % Variance_score)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
model = DecisionTreeRegressor()
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
Variance_score = model.score(X_test_fs, y_test)
print('Variance score for top features: %.2f' % Variance_score)

#XGBRegressor(n_jobs=30) testing
# fit the model
print('XGBRegressor traning and evaluation')
model = XGBRegressor(n_jobs=30)
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
Variance_score = model.score(X_test, y_test)
print('Variance score for full features: %.2f' % Variance_score)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
model = XGBRegressor(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
Variance_score = model.score(X_test_fs, y_test)
print('Variance score for top features: %.2f' % Variance_score)


#RFE Recursive feature elimination to check the feature ranking
from sklearn.feature_selection import RFE
#Feature ranking by RFE
model = RandomForestRegressor(n_jobs=30)
rfe = RFE(model, 9)
rfe = rfe.fit(df[inp1], df[outcome])
for i in range(df[inp1].shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
print(rfe.support_)
print(rfe.ranking_)


# evaluate RFE for regression
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
# create pipeline
rfe = RFE(estimator=XGBRegressor(n_jobs=30), n_features_to_select=15)
model = XGBRegressor(n_jobs=30)
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('evaluate RFE for regression')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))



'''
# explore the number of selected features for RFE
# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(2, 20):
		rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=i)
		model = DecisionTreeRegressor()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.savefig('RFEBOX.png')


# automatically select the number of features for RFE
from sklearn.feature_selection import RFECV
# create pipeline
rfe = RFECV(estimator=DecisionTreeRegressor())
model = DecisionTreeRegressor()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# explore the algorithm wrapped by RFE
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingRegressor
 

# get a list of models to evaluate
def get_models():
	models = dict()
	# lr
	rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['lr'] = Pipeline(steps=[('s',rfe),('m',model)])
	# perceptron
	rfe = RFE(estimator=Perceptron(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['per'] = Pipeline(steps=[('s',rfe),('m',model)])
	# cart
	rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['cart'] = Pipeline(steps=[('s',rfe),('m',model)])
	# rf
	rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['rf'] = Pipeline(steps=[('s',rfe),('m',model)])
	# gbm
	rfe = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['gbm'] = Pipeline(steps=[('s',rfe),('m',model)])
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	return scores
 
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.savefig('RFEBOXall.png')
'''
