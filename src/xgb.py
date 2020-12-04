import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Read data
iowa_training_dataset = pd.read_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/train.csv', index_col='Id')
iowa_test_dataset = pd.read_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/test.csv', index_col='Id')

# Drop rows with empty targets, separate target from dataset
iowa_training_dataset.dropna(axis=0, subset=['SalePrice'], inplace=True)
iowa_prices = iowa_training_dataset.SalePrice
iowa_training_dataset.drop(['SalePrice'], axis=1, inplace=True)

# Break of training and validation data
iowa_train_data, iowa_validate_data, iowa_train_prices, iowa_validate_prices = train_test_split(iowa_training_dataset, iowa_prices, train_size=0.8, test_size=0.2, random_state=0)

# Select columns with relatively low cardinality
low_cardinality_cols = [col for col in iowa_train_data.columns if iowa_train_data[col].nunique() < 10 and iowa_train_data[col].dtype == 'object']

# Select numerical columns
numerical_cols = [col for col in iowa_train_data.columns if iowa_train_data[col].dtype in ['int64', 'float64']]

# Keep selected columns
selected_cols = low_cardinality_cols + numerical_cols
iowa_train_data = iowa_train_data[selected_cols].copy()
iowa_validate_data = iowa_validate_data[selected_cols].copy()
iowa_test_dataset = iowa_test_dataset[selected_cols].copy()

# One hot encode
iowa_train_data = pd.get_dummies(iowa_train_data)
iowa_validate_data = pd.get_dummies(iowa_validate_data)
iowa_test_dataset = pd.get_dummies(iowa_test_dataset)
iowa_train_data, iowa_validate_data = iowa_train_data.align(iowa_validate_data, join='left', axis=1)
iowa_train_data, iowa_test_dataset = iowa_train_data.align(iowa_test_dataset, join='left', axis=1)

# XGBoost at default settings
iowa_model_1 = XGBRegressor(random_state=0)
iowa_model_1.fit(iowa_train_data, iowa_train_prices)
predictions_1 = iowa_model_1.predict(iowa_validate_data)
print("Mean Absolute Error at default settings: ", mean_absolute_error(iowa_validate_prices, predictions_1))

# XGBoost at optimized settings
iowa_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
iowa_model_2.fit(iowa_train_data, iowa_train_prices, early_stopping_rounds=5, eval_set=[(iowa_validate_data, iowa_validate_prices)], verbose=False)
predictions_2 = iowa_model_2.predict(iowa_validate_data)
print("Mean Absolute Error at optimized settings: ", mean_absolute_error(iowa_validate_prices, predictions_2))

# Get test result
test_predictions = iowa_model_2.predict(iowa_test_dataset)
output = pd.DataFrame({'Id': iowa_test_dataset.index,
                       'SalePrice': test_predictions})
output.to_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Submissions/boosted.csv', index=False)
