import pandas as pd
import numpy as np
import math
from sklearn import linear_model

# ======================================================
# PART 1: HOME PRICES (Multivariable Linear Regression)
# ======================================================

df = pd.read_csv(r"C:\ML\homeprices.csv")

# Fill missing bedrooms with median
median_bedrooms = math.floor(df['bedrooms'].median())
df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)

# Train model
reg_home = linear_model.LinearRegression()
reg_home.fit(df[['area', 'bedrooms', 'age']], df['price'])

# Prediction
home_price_prediction = reg_home.predict([[2500, 5, 2]])
print("Home price prediction:", home_price_prediction)

print("Home coefficients:", reg_home.coef_)
print("Home intercept:", reg_home.intercept_)

# ======================================================
# PART 2: HIRING DATASET (Salary Prediction)
# ======================================================

df = pd.read_csv(r"C:\ML\hiring.csv")

# Remove hidden spaces in column names
df.columns = df.columns.str.strip()

# Convert experience words to numbers
word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3,
    'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11
}

df['experience'] = df['experience'].replace(word_to_num)

# Fill missing experience with median
median_experience = df['experience'].median()
df['experience'] = df['experience'].fillna(median_experience)

# Fill missing test scores with median
median_test_score = math.floor(df['test_score(out of 10)'].median())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)

# Train model
reg_hiring = linear_model.LinearRegression()
reg_hiring.fit(
    df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']],
    df['salary($)']
)

# Prediction
salary_prediction = reg_hiring.predict([[2, 9, 6]])
print("Predicted salary:", salary_prediction)
