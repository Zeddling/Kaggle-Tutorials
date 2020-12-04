import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

iowa_train_data = pd.read_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/train.csv')
iowa_test_data = pd.read_csv('/home/zeddling/Documents/Personal/KaggleTutorials/Input/iowa/test.csv')

iowa_train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
iowa_prices = iowa_train_data.SalePrice
iowa_train_data.drop(['SalePrice'], axis=1, inplace=True)

numeric_cols = [col for col in iowa_train_data.columns if iowa_train_data[col].dtype in ['int64', 'float64']]
iowa_data = iowa_train_data[numeric_cols].copy()
iowa_test = iowa_test_data[numeric_cols].copy()


def get_score(n_estimators):
    pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])
    score = -1 * cross_val_score(pipeline, iowa_data, iowa_prices, cv=3, scoring='neg_mean_absolute_error')
    return score.mean()


results = dict()
for i in range(1, 9):
    results[i * 50] = get_score(i * 50)

plt.plot(list(results.keys()), list(results.values()))
plt.show()

max(results)
