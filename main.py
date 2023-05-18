import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


strategy_array = [
    LinearRegression,  # 0
    BayesianRidge,  # 1
    MLPRegressor,  # 2
    GaussianProcessRegressor,  # 3
    DecisionTreeRegressor,  # 4
    KNeighborsRegressor,  # 5
    RandomForestRegressor,  # 6
    GradientBoostingRegressor  # 7
]


print('Loading...')
# digits = datasets.load_digits()
data = pd.read_csv('./melb_data.csv')
X = data[[
    'Rooms',
    'Bedroom2',
    'Bathroom',
    'Car',
    'Landsize',
    'BuildingArea',
    'Lattitude',
    'Longtitude',
    'Distance'
]].copy()

y = data['Price']

encoder = LabelEncoder()

X['Suburb_encoded'] = encoder.fit_transform(data['Suburb'])
X['Address_encoded'] = encoder.fit_transform(data['Address'])
X['Regionname_encoded'] = encoder.fit_transform(data['Regionname'])
X['Type_encoded'] = encoder.fit_transform(data['Type'])
X['Method_encoded'] = encoder.fit_transform(data['Method'])

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

print('Splitting...')
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.2,
    random_state=42
)

try:
    choice = input('Choose strategy (0-7)\n')
    choice = int(choice)
    if choice > 7:
        choice = 0
except ValueError:
    choice = 0


method = strategy_array[choice]
print(f'Using {method.__name__} strategy')
model = method()

print('Fitting...')
model.fit(X_train, y_train)

print('Predicting...')
predictions = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Model Score", accuracy)
