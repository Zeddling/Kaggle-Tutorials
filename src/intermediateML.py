import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Read the data
print("Missing Values")
iowa_full = pd.read_csv("/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/train.csv", index_col="Id")
iowa_full_test = pd.read_csv("/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/test.csv", index_col="Id")

# remove rows with missing targets, separate targets from predictors
iowa_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
prices = iowa_full.SalePrice
iowa_full.drop(['SalePrice'], axis=1, inplace=True)

# Use numerical predictors
iowa_data = iowa_full.select_dtypes(exclude=['object'])
iowa_data_test = iowa_full_test.select_dtypes(exclude=['object'])

# Break off validation set from training data
iowa_training_data, iowa_validation_data, iowa_training_prices, iowa_validation_prices = train_test_split(iowa_data,
                                                                                                          prices,
                                                                                                          train_size=0.8,
                                                                                                          test_size=0.2,
                                                                                                          random_state=0)

# Diagnose data for missing values: 1. Shape of training data, 2. number of missing values
print(iowa_training_data.shape)

missing_values_count_by_column = (iowa_training_data.isnull().sum())
print(missing_values_count_by_column[missing_values_count_by_column > 0])


# Data diagnostics show that less than 25% of the data is missingx

# Create data score_dataset() function
def score_dataset(X_train, X_validate, y_train, y_validate):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_validate)
    return mean_absolute_error(y_validate, prediction)


# Approach of removing columns with null values
cols_with_missing_values = [col for col in iowa_training_data.columns if iowa_training_data[col].isnull().any()]
reduced_training_data = iowa_training_data.drop(cols_with_missing_values, axis=1)
reduced_validation_data = iowa_validation_data.drop(cols_with_missing_values, axis=1)

print("Mean Absolute Error by dropping columns: ",
      score_dataset(reduced_training_data, reduced_validation_data, iowa_training_prices, iowa_validation_prices))

# Approach by simple imputation using median strategy
my_imputer = SimpleImputer(strategy="median")

imputed_training_data = pd.DataFrame(my_imputer.fit_transform(iowa_training_data))
imputed_validation_data = pd.DataFrame(my_imputer.transform(iowa_validation_data))

imputed_training_data.columns = iowa_training_data.columns
imputed_validation_data.columns = iowa_validation_data.columns

print("Mean Absolute Error by simple imputation: ",
      score_dataset(imputed_training_data, imputed_validation_data, iowa_training_prices, iowa_validation_prices))

# Final predictions using median strategy on imputer
final_imputer = SimpleImputer(strategy="median")
final_iowa_training_data = pd.DataFrame(final_imputer.fit_transform(iowa_training_data))
final_iowa_validation_data = pd.DataFrame(final_imputer.transform(iowa_validation_data))

final_iowa_training_data.columns = iowa_training_data.columns
final_iowa_validation_data.columns = iowa_validation_data.columns

final_iowa_model = RandomForestRegressor(n_estimators=100, random_state=0)
final_iowa_model.fit(final_iowa_training_data, iowa_training_prices)
final_iowa_prices_predictions = final_iowa_model.predict(final_iowa_validation_data)

print("Final Mean Absolute Error: ", mean_absolute_error(iowa_validation_prices, final_iowa_prices_predictions))

# Use test data to submit good results; use already trained model and iowa_data_test(has no dtype 'object') to
# generate prediction
final_iowa_test_data = pd.DataFrame(final_imputer.transform(iowa_data_test))
iowa_test_predictions = final_iowa_model.predict(final_iowa_test_data)

# Save predictions in file
# output = pd.DataFrame({'Id': iowa_data_test.index,
#                      'SalePrice': iowa_test_predictions})
# output.to_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Submissions/submission.csv', index=False)
