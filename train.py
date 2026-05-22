import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# load the dataset
churn = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Encode the categorical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in churn.columns:
    if churn[column].dtype == 'object':
        churn[column] = le.fit_transform(churn[column])

churn['TotalCharges'] = churn['TotalCharges'].replace(' ', np.nan)
churn['TotalCharges'] = pd.to_numeric(churn['TotalCharges'])
churn['TotalCharges'].fillna(churn['TotalCharges'].median(), inplace=True)


scaler = StandardScaler()

churn[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(churn[['MonthlyCharges', 'TotalCharges', 'tenure']])

customer_ids = churn['customerID']

x = churn.drop(['customerID', 'Churn'], axis = 1)
y = churn['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
# fit data into model
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Confusion Matrix
print(confusion_matrix(y_test, y_pred))
# Classification Report
print(classification_report(y_test, y_pred))
# Save model
pickle.dump(model, open("model.pkl", "wb"))
# Save Scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model trained successfully!")
