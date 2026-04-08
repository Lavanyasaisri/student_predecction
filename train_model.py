import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_excel('dataset/student_performance_data.xlsx')

# Drop identifiers and derived columns
data = data.drop(['Roll_No', 'Name', 'Pass_Fail'], axis=1)

# Encode categorical variables
categorical_cols = ['Gender', 'Major', 'PartTimeJob', 'ExtraCurricularActivities', 'Grade']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Features and target
X = data.drop('Final_Score', axis=1)
y = data['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, predictions)}')

# Save model
joblib.dump(model, 'models/student_model.pkl')