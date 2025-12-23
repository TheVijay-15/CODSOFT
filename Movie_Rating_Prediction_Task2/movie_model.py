import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. load data
df = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/CODSOFT/Movie_Rating_Prediction_Task2/IMDb Movies India.csv", encoding='latin1')

# 2. data cleaning
# drop rows missing rating 
df.dropna(subset=['Rating'], inplace=True)

# Clean Year and Duration to numeric
df['Year'] = df['Year'].str.extract('(\d+)').astype(float)
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)
df['Duration'].fillna(df['Duration'].median(), inplace=True)

# Handle Votes (remove commas and convert to float)
df['Votes'] = df['Votes'].str.replace(',', '').astype(float)
df['Votes'].fillna(df['Votes'].median(), inplace=True)

# 3. feature engineering
# Mean Rating per category
for col in ['Genre', 'Director', 'Actor 1']:
    df[col + '_encoded'] = df.groupby(col)['Rating'].transform('mean')

# Fill new encoded columns 
df.fillna(df['Rating'].mean(), inplace=True)

# 4. modeling
features = ['Year', 'Duration', 'Votes', 'Genre_encoded', 'Director_encoded', 'Actor 1_encoded']
X = df[features]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. results
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Test Prediction
print("\nSample Prediction for a movie:")
print(f"Actual: {y_test.iloc[0]}, Predicted: {y_pred[0]:.1f}")