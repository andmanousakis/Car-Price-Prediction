import pandas as pd
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

def get_dataset(dataset_name):

	if dataset_name == "data":
		data = pd.read_csv('data.csv')
	else:
		data = "Select a dataset from the dropdown to the left."
	
	return data

def OHE(x, df):
    
	temp = pd.get_dummies(df[x], drop_first = True)
	df = pd.concat([df, temp], axis = 1)
	df.drop([x], axis = 1, inplace = True)
	
	return df

def add_parameter_ui(regressor_name):

	params = dict()
	if regressor_name == "Extra Trees Regressor":

		n_estimators = st.sidebar.slider("Nr. of estimators:", 1, 200)
		random_state = st.sidebar.slider("Random state:", 0, 100)
		#min_samples_leaf = st.sidebar.slider("Min samples leaf:", 1, 10)
		#min_samples_split = st.sidebar.slider("Min samples split", 1, 10)
		cv = st.sidebar.slider("Folds:", 1, 10)
		training_size = st.sidebar.slider("Training size:", 0.1, 1.0)
		scoring = st.sidebar.selectbox("Select loss metric:", ("neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error", "r2"))
		params = {"n_estimators": n_estimators,
			"random_state": random_state,
			#"min_samples_leaf": min_samples_leaf,
			#"min_samples_split": min_samples_split,
			"scoring": scoring,
			"Folds": cv,
			"training_size": training_size}

	return params

def get_regressor(regressor_name, params, dataOHE):

	if regressor_name == "Extra Trees Regressor":

		regressor = ExtraTreesRegressor(n_estimators = params["n_estimators"],
										random_state = params["random_state"])
		
		return regressor

def optimize_model(model, params, X_train, y_train):

	gsc = GridSearchCV(
	    estimator = model,
	    param_grid = {
	        'n_estimators': [150, 100, 70],
	        #'max_features': range(5, 8, 10),
	        'min_samples_leaf': [10, 4, 3, 2, 1],
	        'min_samples_split': [10, 8, 6],
	    },
	    scoring = params["scoring"],
	    cv = params["Folds"]
	)

	# Training the model on every parameter combination.
	grid_result = gsc.fit(X_train, y_train)
	st.write("R2 score:", abs(grid_result.best_score_),"Best parameters:", grid_result.best_params_)
	#print("Best score on training data: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))

	# Re-training the model with optimal parameters.
	tuned_etr = ExtraTreesRegressor(**grid_result.best_params_)

	return tuned_etr