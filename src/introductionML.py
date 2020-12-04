#Explore your data
import pandas as pd

#Build model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeRegressor

#File path of file to read
melbourne_file_path = "/home/zeddling/Documents/Personal/KaggleTutorials/Input/melbourne/melb_data.csv"

#read data and store data in a DataFrame
melbourne_data = pd.read_csv(melbourne_file_path)

#print summary of the data in Melbourne data
#print(melbourne_data.describe())

#####  Using dot notation   ##### Selecting prediction target
prices = melbourne_data.Price


#####  Choosing features    #####
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
melbourne_dataToBe_Predicted = melbourne_data[melbourne_features]

#Define model. random stats set two one for constant results each run
melbourne_model = DecisionTreeRegressor(random_state = 1)

melbourne_model.fit(melbourne_dataToBe_Predicted, prices)

print("Making predictions for the following 5 houses:")
print(melbourne_dataToBe_Predicted.head())
print("The predictions are")
print(melbourne_model.predict(melbourne_dataToBe_Predicted.head()))

#Get MAE
predicted_prices = melbourne_model.predict(melbourne_dataToBe_Predicted)
mae = mean_absolute_error(prices, predicted_prices)
print("Mean Absolute Error: ",mae)

#Proper Model Validation
train_melbourne_dataToBe_Predicted, val_melbourne_dataToBe_Predicted, train_prices, val_prices = train_test_split(melbourne_dataToBe_Predicted, prices, random_state=0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_melbourne_dataToBe_Predicted, train_prices)

val_predictions = melbourne_model.predict(val_melbourne_dataToBe_Predicted)
mae = mean_absolute_error(val_prices, val_predictions)
print("Actual MAE: ", mae)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

best_mae = get_mae(5, train_melbourne_dataToBe_Predicted, val_melbourne_dataToBe_Predicted, train_prices, val_prices)
best_tree_size = 5
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_melbourne_dataToBe_Predicted, val_melbourne_dataToBe_Predicted, train_prices, val_prices)
    print("Max leaf nodes: %d \t\tMean Absolute Error: %d"%(max_leaf_nodes, my_mae))

    if my_mae < best_mae:
        best_tree_size = max_leaf_nodes
        best_mae = my_mae


print("Best tree size: ", best_tree_size)

## ............Random Forest................##
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_melbourne_dataToBe_Predicted, train_prices)
melbourne_preds = forest_model.predict(val_melbourne_dataToBe_Predicted)
print("Random Forest model predictions for Melbourne: ",mean_absolute_error(val_prices, melbourne_preds))