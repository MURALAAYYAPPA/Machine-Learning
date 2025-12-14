import numpy as np 
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\income.csv")
plt.scatter(df.year,df.income)
plt.show()
plt.xlabel('year')
plt.ylabel('income')
reg=linear_model.LinearRegression()
reg.fit(df[['year']],df.income)
reg.predict([[2020]])
reg.coef_
reg.intercept_
plt.scatter(df.year,df.income)
plt.plot(df.year,reg.predict(df[['year']]))
plt.show()
