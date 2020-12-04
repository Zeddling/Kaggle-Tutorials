import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

iowa_dataset = pd.read_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/train.csv', index_col='Id')
iowa_test_dataset = pd.read_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/test.csv', index_col='Id')

iowa_dataset.dropna(axis=0, subset=['SalePrice'], inplace=True)
iowa_prices = iowa_dataset.SalePrice
iowa_dataset.drop(['SalePrice'], axis=1, inplace=True)

iowa_training_dataset, iowa_validation_dataset, iowa_training_prices, iowa_validation_prices = train_test_split(
    iowa_dataset, iowa_prices, train_size=0.8, test_size=0.2, random_state=0)

# Select categorical columns with relatively low cardinality
categorical_cols = [col for col in iowa_training_dataset.columns if
                    iowa_training_dataset[col].nunique() < 10 and iowa_training_dataset[col].dtype == 'object']

# Select numerical columns
numerical_cols = [col for col in iowa_training_dataset.columns if
                  iowa_training_dataset[col].dtype in ['int64', 'float64']]

# Keep selected columns only
selected_cols = categorical_cols + numerical_cols
iowa_training_dataset = iowa_training_dataset[selected_cols].copy()
iowa_validation_dataset = iowa_validation_dataset[selected_cols].copy()
iowa_test_dataset = iowa_test_dataset[selected_cols].copy()

# Preprocess numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocess for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

iowa_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', iowa_model)
                              ])

# Preprocessing of training data, fit model
my_pipeline.fit(iowa_training_dataset, iowa_training_prices)

# Preprocessing of validation data, get predictions
iowa_prediction_prices = my_pipeline.predict(iowa_validation_dataset)

# Evaluate the model
score = mean_absolute_error(iowa_validation_prices, iowa_prediction_prices)
print('Mean Absolute Error:', score)

# Preprocess of test data
predict_test = my_pipeline.predict(iowa_test_dataset)

output = pd.DataFrame({
    'Id': iowa_test_dataset.index,
    'SalePrice': predict_test
})
output.to_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Submissions/pipelines.csv', index=False)