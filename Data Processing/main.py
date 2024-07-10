import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('random_dataframe.csv')

# AVE = df['a'][3:8].mean()
# df['a'] = df['a'] - AVE
# df_subset = df.iloc[3:]


def fit_polynomial_regression(row, degree=2):
    X = np.arange(len(row)).reshape(-1, 1)
    y = row.values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return y_pred

df_transformed = df.copy()
for index, row in df.iterrows():
    df_transformed.loc[index, ['a', 'b', 'c']] = fit_polynomial_regression(row[['a', 'b', 'c']])


# Display the subset DataFrame
print(df_transformed)
