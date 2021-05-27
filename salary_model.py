import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

db = pd.read_csv('salary.csv')
x = db["YearsExperience"].values.reshape(-1,1)  
y = db['Salary']

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=42)

model.fit(X_train, y_train)
model.coef_

joblib.dump(model, 'salary.pkl')
