import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("--- Stage 2: Data Preparation ---")
try:
    transaction_data = pd.read_csv('customer_transactions.csv')
    support_data = pd.read_csv('support_tickets.csv')
    online_data = pd.read_csv('online_interactions.csv')
    churn_status = pd.read_csv('customer_churn_status.csv')
    print("Data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

customer_data = pd.merge(churn_status, transaction_data, on='CustomerID', how='left')
customer_data = pd.merge(customer_data, support_data, on='CustomerID', how='left')
customer_data = pd.merge(customer_data, online_data, on='CustomerID', how='left')

print("\nMissing values before imputation:\n", customer_data.isnull().sum())
numerical_cols = customer_data.select_dtypes(include=np.number).columns.tolist()
if 'Churned' in numerical_cols:
    numerical_cols.remove('Churned')
imputer_numerical = SimpleImputer(strategy='mean')
customer_data[numerical_cols] = imputer_numerical.fit_transform(customer_data[numerical_cols])

categorical_cols = customer_data.select_dtypes(include='object').columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
customer_data[categorical_cols] = imputer_categorical.fit_transform(customer_data[categorical_cols])
print("\nMissing values after imputation:\n", customer_data.isnull().sum())
customer_data.drop_duplicates(inplace=True)
customer_data['RegistrationDate'] = pd.to_datetime(customer_data['RegistrationDate'], errors='coerce')
customer_data['TenureMonths'] = ((pd.to_datetime('now') - customer_data['RegistrationDate']).dt.days) // 30
customer_data['TotalSpending'] = customer_data['PurchaseAmount'].fillna(0)
columns_to_drop = ['TransactionDate', 'TicketDate', 'InteractionDate', 'RegistrationDate']
customer_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

categorical_features = customer_data.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(customer_data[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
customer_data = pd.concat([customer_data.drop(categorical_features, axis=1).reset_index(drop=True), encoded_df], axis=1)

numerical_features = customer_data.select_dtypes(include=np.number).columns.tolist()
if 'Churned' in numerical_features:
    numerical_features.remove('Churned')
scaler = StandardScaler()
customer_data[numerical_features] = scaler.fit_transform(customer_data[numerical_features])

X = customer_data.drop(['CustomerID', 'Churned'], axis=1).select_dtypes(include=np.number).dropna(axis=1)
y = customer_data['Churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


print("\nPrepared data sample:")
print(X_train.head())
print("\nShape of training features:", X_train.shape)
print("Shape of testing features:", X_test.shape)

print("\n--- Stage 3: Modeling ---")
logistic_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
logistic_model.fit(X_train, y_train)
print("\nTrained Logistic Regression model:", logistic_model)

param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=2, scoring='f1')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
print("\nBest Random Forest Model:", best_rf_model)

print("\n--- Stage 4: Data Analytics ---")

plt.figure(figsize=(8, 6))
sns.countplot(x=y_train)
plt.title('Distribution of Churn in Training Data')
plt.xlabel('Churn Status (0: No, 1: Yes)')
plt.ylabel('Number of Customers')
plt.show()

if 'best_rf_model' in locals():
    feature_importances = best_rf_model.feature_importances_
    feature_names = X_train.columns
    sorted_indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances[sorted_indices[:10]], y=feature_names[sorted_indices[:10]])
    plt.title('Top 10 Feature Importance in Best Random Forest Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.show()

print("\n--- Stage 5: Deployment ---")
if 'best_rf_model' in locals():
    model_filename = 'churn_prediction_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(best_rf_model, file)
    print(f"\nTrained Random Forest model saved as {model_filename}")

loaded_model = None
try:
    with open('churn_prediction_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    print("\nModel loaded successfully!")
except FileNotFoundError:
    print("\nModel file not found.")

if loaded_model is not None and not X_test.empty:
    sample_new_data = X_test.head()
    predictions = loaded_model.predict(sample_new_data)
    print("\nSample predictions for new data:\n", predictions)
elif X_test.empty:
    print("\nNo test data available for prediction.")

print("\n--- Stage 6: Evaluation ---")
if loaded_model is not None and not X_test.empty:
    y_pred = loaded_model.predict(X_test)
    y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\nEvaluation Metrics on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
else:
    print("\nModel not loaded or test data is empty, cannot evaluate.")
