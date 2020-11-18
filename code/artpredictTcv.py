#Importing libraries and the data set:
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_validate,TimeSeriesSplit,RepeatedKFold,cross_val_predict
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.pipeline import Pipeline
from numpy import mean,std
import warnings
warnings.filterwarnings("ignore")
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.svm import SVR
from sknn.mlp import Regressor, Layer



#load the inoput from the file
df = pd.read_csv("./df_for_ml_stata.csv", low_memory=False) #Reading the dataset in a dataframe using Pandas

#summary of numerical fieldsprint(df.head(10))
print(df.describe())

#Check missing values in the dataset
df.apply(lambda x: sum(x.isnull()), axis=0)

df.replace(np.nan,0, inplace=True)

#Building a Predictive Model
outcome = ['price_usd']
predictors = df.columns.difference(['price_usd'])

#setting input and ouput for the model
inp1 = df.columns
X1=df[predictors]
y=df[outcome].values.ravel()

# normalising/scaling all the values for optimisation
scaler = StandardScaler()
X = scaler.fit_transform( X1 )

# evaluation of a model using all features
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
# evaluate a give model using cross-validation
tscv = TimeSeriesSplit(n_splits = 5)
def evaluate_model(model, X, y):
	tscv = TimeSeriesSplit(n_splits = 5)
	scores = cross_validate(model, X, y, scoring=('r2', 'neg_mean_squared_error'), cv=tscv,return_train_score=True, n_jobs=-1, error_score='raise')
	return scores
# fit the model
print('RandomForestRegressor with full features\n')
model = RandomForestRegressor(n_jobs=30)
scores = evaluate_model(model, X_train, y_train)
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')



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


# feature selection
model = RandomForestRegressor(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
scores = evaluate_model(model, X_train_fs, y_train)
print('RandomForestRegressor with top features\n')
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train_fs, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test_fs, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

#LinearRegression testing
# fit the model
print('LinearRegression with full features\n')
model = LinearRegression(n_jobs=30)
scores = evaluate_model(model, X_train, y_train)
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

# feature selection
model = LinearRegression(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
scores = evaluate_model(model, X_train_fs, y_train)
# evaluate predictions
print('LinearRegression with top features\n')
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train_fs, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test_fs, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

#DecisionTreeRegressor() testing
# fit the model
print('DecisionTreeRegressor with full features\n')
model = DecisionTreeRegressor()
scores = evaluate_model(model, X_train, y_train)
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

# feature selection
model = DecisionTreeRegressor()
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
scores = evaluate_model(model, X_train_fs, y_train)
# evaluate predictions
print('DecisionTreeRegressor with top features\n')
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train_fs, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test_fs, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

#XGBRegressor(n_jobs=30) testing
# fit the model
print('XGBRegressor with full features\n')
model = XGBRegressor(n_jobs=30)
scores = evaluate_model(model, X_train, y_train)
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

# feature selection
model = XGBRegressor(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
scores = evaluate_model(model, X_train_fs, y_train)
# evaluate predictions
print('XGBRegressor with top features\n')
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train_fs, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test_fs, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

#AdaBoostRegressor(n_jobs=30) testing
# fit the model
print('AdaBoostRegressor with full features\n')
model = AdaBoostRegressor()
scores = evaluate_model(model, X_train, y_train)
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

# feature selection
model = AdaBoostRegressor()
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
scores = evaluate_model(model, X_train_fs, y_train)
# evaluate predictions
print('AdaBoostRegressor with top features\n')
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train_fs, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test_fs, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')


#svm_regressor training and  testing
# fit the model
print('svm_regressor with full features\n')
model = SVR(kernel = 'rbf',C=10,gamma=1)
scores = evaluate_model(model, X_train, y_train)
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

# feature selection
model = RandomForestRegressor(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
model = SVR(kernel = 'rbf',C=10,gamma=1)
scores = evaluate_model(model, X_train_fs, y_train)
# evaluate predictions
print('svm_regressor with top features\n')
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train_fs, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test_fs, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')


#neural_net training and  testing
# fit the model
print('neural_net with full features\n')
model = Regressor(layers=[Layer("Rectifier", units=500),  # Hidden Layer1
                        Layer("Rectifier", units=300)  # Hidden Layer2
    , Layer("Linear")],  # Output Layer
                n_iter=100, learning_rate=0.01)

scores = evaluate_model(model, X_train, y_train)
# evaluate predictions
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')


# feature selection
model = RandomForestRegressor(n_jobs=30)
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,model)
# fit the model
model = Regressor(layers=[Layer("Rectifier", units=500),  # Hidden Layer1
                        Layer("Rectifier", units=300)  # Hidden Layer2
    , Layer("Linear")],  # Output Layer
                n_iter=100, learning_rate=0.01)

scores = evaluate_model(model, X_train_fs, y_train)
# evaluate predictions
print('neural_net with top features\n')
print('MSE: %.3f (%.3f)' % (mean(scores['test_neg_mean_squared_error']), std(scores['test_neg_mean_squared_error'])))
print(scores['test_neg_mean_squared_error'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
print('R2: %.3f (%.3f)' % (mean(scores['train_r2']), std(scores['train_r2'])))
print(scores['train_r2'])
print("Loss: {0:.3f} (+/- {1:.3f})".format(scores['train_r2'].mean(), scores['train_r2'].std()))
# evaluate predictions
training_predictions = cross_val_predict(model, X_train_fs, y_train, cv=tscv.n_splits)
testing_predictions = cross_val_predict(model, X_test_fs, y_test, cv=tscv.n_splits)
training_accuracy = metrics.r2_score(y_train,training_predictions)
test_accuracy = metrics.r2_score(y_test,testing_predictions)
print("training-predictions accuracy:",training_accuracy)
print("Test-predictions accuracy:",test_accuracy,'\n')

