import joblib

print("---------WELCOME TO SALARY PREDICTOR---------")
exp = float(input("Enter your experience")
model = joblib.load('salary.pkl')
            
ypred=model.predict([[exp]])
print("Salary: ", int(ypred))
            
