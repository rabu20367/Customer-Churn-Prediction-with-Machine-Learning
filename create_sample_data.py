import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
n_samples = 100

data = {
    'customerID': [f'CUST{i:03d}' for i in range(n_samples)],
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.randint(0, 2, n_samples),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples),
    'tenure': np.random.randint(1, 72, n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
    'MonthlyCharges': np.random.uniform(20, 120, n_samples),
    'TotalCharges': np.random.uniform(20, 8000, n_samples),
    'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
}

df = pd.DataFrame(data)
df.to_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)
print('Sample dataset created successfully!')
print(f'Dataset shape: {df.shape}')
print(f'Churn rate: {df["Churn"].value_counts(normalize=True)}')
