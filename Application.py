# Libraries
import streamlit as st
import plotly_express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functions import *

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

# Title of the app.
st.title("Car Price Prediction")

# Sidebar.
#st.sidebar.subheader("Settings")

### Load the dataset. ###

# Select a dataset.
dataset_name = st.sidebar.selectbox("Select Dataset:", ("None", "SpotaWheel"))

# Select a regressor.
regressor_name = st.sidebar.selectbox("Select Regressor:", ("None", "Extra Trees Regressor"))

data = get_dataset(dataset_name)

# If a dataset has not been selected, show a message.
if not isinstance(data, pd.DataFrame):

	st.write("Select a dataset from the dropdown to the left.")


### Preparing the dataset ###
if dataset_name == 'SpotaWheel':

	# Get the car brand from the variable 'Name'.
	if 'Name' in data:
	    Brand = data['Name'].apply(lambda x : x.split(' ')[0])
	    data.insert(3, "Brand", Brand)
	    data.drop(['Name'],axis = 1,inplace = True)
	else:
		pass

	# Encode the categorical variables 'Fuel_Type', 'Transmission', 'Brand'
	dataOHE = OHE('Fuel_Type', data)
	dataOHE = OHE('Transmission', dataOHE)
	dataOHE = OHE('Brand', dataOHE)

	col1, col2 = st.columns(2)
	col1.write(dataOHE); col1.write(dataOHE.shape)
	col2.write(dataOHE.describe())

if regressor_name == 'Extra Trees Regressor':

	# Model parameters.
	params = add_parameter_ui(regressor_name)

	# Regression.
	np.random.seed(0)
	X = dataOHE.drop(['Price'], axis = 1)
	y = dataOHE['Price']

	X_train, X_test, y_train, y_test = train_test_split(X, y, 
	                                                    train_size = params["training_size"],
	                                                    test_size = 1 - params["training_size"],
	                                                    random_state = 100)
	features = ['Kilometers_Driven', 'Owner_Type', 'Mileage', 'Engine', 'Power',
   'Seats', 'Age', 'Diesel', 'LPG', 'Petrol', 'Manual']
	X_train_basic = X_train[features]
	fig = plt.figure(figsize = (15, 10))
	sns.heatmap(X_train_basic.corr(), annot = True, cmap="YlGnBu")
	col1.write(fig)

	import ppscore as pps

	X_train_basic = X_train[features]
	fig = plt.figure(figsize = (15, 10))
	ppsMatrix = pps.matrix(X_train_basic).pivot(columns = 'x', index = 'y',  values = 'ppscore')
	sns.heatmap(ppsMatrix, annot = True)
	col2.write(fig)

	regressor = get_regressor(regressor_name, params, dataOHE)

if st.sidebar.button('Run'):

	# Training the model.
	regressor.fit(X_train, y_train)

	# Grid search to optimize the model.
	tuned_regressor = optimize_model(regressor, params, X_train, y_train)
	tuned_regressor.fit(X_train, y_train)

	# Testing the optimized model.
	y_pred = tuned_regressor.predict(X_test)

	# Residuals plot. -> Not supported in streamlit yet.
	'''from yellowbrick.regressor import ResidualsPlot

	visualizer = ResidualsPlot(tuned_regressor)
	visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
	visualizer.score(X_test, y_test)  # Evaluate the model on the test data
	#visualizer.show()                 # Finalize and render the figure
	col1.write(visualizer)'''