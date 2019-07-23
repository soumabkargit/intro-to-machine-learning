import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Path of the file to read
iowa_file_path = 'input/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Use the command you learned to view summary statistics of the data. Then fill in variables to answer the following questions
home_data.describe()

# print the list of columns in the dataset to find the name of the prediction target
home_data.columns

y = home_data.SalePrice

# Create the list of features below
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

# select data corresponding to features in feature_names
X = home_data[feature_names]

# Review data
# print description or statistics from X
X.describe()

# print the top few lines
print(X.head)

#specify the model.
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X,y)

predictions = iowa_model.predict(X)
print(predictions)

# fll in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
# fit your model
rf_model.fit(X, y)
# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = rf_model.predict(X)
print(rf_val_mae)