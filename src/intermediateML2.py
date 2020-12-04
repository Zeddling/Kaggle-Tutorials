import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print("Handling categorical data")
iowa_full = pd.read_csv("/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/train.csv", index_col="Id")
iowa_test_data = pd.read_csv("/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/test.csv", index_col="Id")

iowa_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
iowa_prices = iowa_full.SalePrice
iowa_full.drop(['SalePrice'], axis=1, inplace=True)

cols_with_missing_values = [col for col in iowa_full.columns if iowa_full[col].isnull().any()]
iowa_full.drop(cols_with_missing_values, axis=1, inplace=True)
iowa_test_data.drop(cols_with_missing_values, axis=1, inplace=True)

iowa_full_training, iowa_full_validation, iowa_prices_training, iowa_prices_validation = train_test_split(iowa_full, iowa_prices, train_size=0.8, test_size=0.2, random_state=0)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    return mean_absolute_error(y_valid, prediction)

#Approach by dropping columns
drop_iowa_train = iowa_full_training.select_dtypes(exclude=['object'])
drop_iowa_valid = iowa_full_validation.select_dtypes(exclude=['object'])

print("Mean Absolute Error by dropping categorical variables: ", score_dataset(drop_iowa_train, drop_iowa_valid, iowa_prices_training, iowa_prices_validation))

#Approach by Label encoding
#print("Unique values in 'Condition2' column in the training data: ", iowa_full_training['Condition2'].unique())
#print("\nUnique values in 'Condition2' column in validation data: ", iowa_full_validation['Condition2'].unique())

#Fix issue of missing columns
object_cols = [col for col in iowa_full_training.columns if iowa_full_training[col].dtype == 'object']

good_label_cols = [col for col in object_cols if set(iowa_full_training[col]) == set(iowa_full_validation[col])]

bad_label_cols = list(set(object_cols) - set(good_label_cols))

#print('Categorical columns that will be label encoded: ', good_label_cols)
#print('\nCategorical columns that will be dropped from the dataset: ', bad_label_cols)

label_iowa_train = iowa_full_training.drop(bad_label_cols, axis=1)
label_iowa_valid = iowa_full_validation.drop(bad_label_cols, axis=1)

label_encoder = LabelEncoder()
for col in set(good_label_cols):
    label_iowa_train[col] = label_encoder.fit_transform(iowa_full_training[col])
    label_iowa_valid[col] = label_encoder.transform(iowa_full_validation[col])

print("Mean Absolute Error by label encoding: ", score_dataset(label_iowa_train, label_iowa_valid, iowa_prices_training, iowa_prices_validation))

#Approach by One Hot Encoding

#Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: iowa_full_training[col].nunique(), object_cols))
object_dict = dict(zip(object_cols, object_nunique))

#print(sorted(object_dict.items(), key=lambda x:x[1]))

low_cardinality_cols = [col for col in object_cols if iowa_full_training[col].nunique() > 10]
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
#print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
#print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

#Apply OH encoding on all columns with categorical data
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(iowa_full_training[low_cardinality_cols]))
oh_cols_valid = pd.DataFrame(oh_encoder.transform(iowa_full_validation[low_cardinality_cols]))

#OH encode removed index; put it back
oh_cols_train.index = iowa_full_training.index
oh_cols_valid.index = iowa_full_validation.index

#Replace categorical columns with one-hot encoding
num_iowa_train = iowa_full_training.drop(object_cols, axis=1)
num_iowa_valid = iowa_full_validation.drop(object_cols, axis=1)

#Add OH encoded columns to numerical features
oh_cols_train = pd.concat([num_iowa_train, oh_cols_train], axis=1)
oh_cols_valid = pd.concat([num_iowa_valid, oh_cols_valid], axis=1)

print("Mean Absolute Error using One Hot Encoding: ", score_dataset(oh_cols_train, oh_cols_valid, iowa_prices_training, iowa_prices_validation))

