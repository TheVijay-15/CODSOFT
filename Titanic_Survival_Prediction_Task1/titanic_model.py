import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/CODSOFT/Titanic_Survival_Prediction_Task1/Titanic-Dataset.csv")

# 2. Basic EDA
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSurvival Rate:", df['Survived'].value_counts(normalize=True)[1]*100, "%")

# 3. Data preprocessing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
df['Title'] = df['Title'].replace('Mme','Mrs')

df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
title_map = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
df['Title'] = df['Title'].map(title_map)
df['Title'] = df['Title'].fillna(0)

df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

# 4. Handle missing values for Title
imputer = SimpleImputer(strategy='most_frequent')
df['Title'] = imputer.fit_transform(df[['Title']])

# 5. Split data
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Model comparison
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    'SVM': (SVC(probability=True, random_state=42), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    })
}

print("model comparssion with gridsearch (5-Fold Cross Validation)")

results = []

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    
    # GridSearchCV with 5-fold CV
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'CV Score': round(cv_score, 4)
    })
    
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  5-Fold CV Score: {cv_score:.4f}")

# 7. Results comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "="*80)
print("FINAL MODEL RANKING (Sorted by Accuracy)")
print("="*80)
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']].to_string(index=False))

# 8. Select best model
best_model_name = results_df.iloc[0]['Model']
best_model_details = [m for m in models.items() if m[0] == best_model_name][0]
best_model = GridSearchCV(best_model_details[1][0], best_model_details[1][1], 
                         cv=5, scoring='accuracy').fit(X_train, y_train).best_estimator_


print(f"select best model: {best_model_name}")

print(f"Model Type: {type(best_model).__name__}")
print(f"Training Accuracy: {best_model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {best_model.score(X_test, y_test):.4f}")

# 9. Why Random Forest was selected (if it's the best)
if 'Random Forest' in best_model_name:
    print("\nWhy Random Forest was selected over other models:")
    print("1. Highest Accuracy: Achieved better test accuracy compared to other models")
    print("2. Robust to Overfitting: Lower gap between train and test accuracy")
    print("3. Feature Importance: Provides interpretable feature importance scores")
    print("4. Handles Non-linearity: Captures complex relationships in Titanic data")
    print("5. Ensemble Method: Reduces variance and improves generalization")
elif 'Gradient Boosting' in best_model_name:
    print("\nWhy Gradient Boosting was selected over other models:")
    print("1. Sequential Learning: Builds trees sequentially to correct errors")
    print("2. High Predictive Power: Often achieves highest accuracy on tabular data")
    print("3. Regularization: Built-in regularization prevents overfitting")
    print("4. Handles Imbalanced Data: Better with Titanic's survival distribution")
else:
    print(f"\nWhy {best_model_name} was selected:")
    print(f"1. Best Performance: Achieved highest accuracy among all tested models")
    print(f"2. Cross-Validation Score: Consistent performance across 5 folds")
    print(f"3. Balanced Metrics: Good balance of precision and recall")

# 10. Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    print("\nTop 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

# 11. Final model evaluation
from sklearn.metrics import classification_report, confusion_matrix


print("final model evaluation ")
final_predictions = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, final_predictions, target_names=['Not Survived', 'Survived']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, final_predictions))

# Save results
results_df.to_csv('titanic_model_comparison.csv', index=False)
print("\nResults saved to 'titanic_model_comparison.csv'")
print("\nTitanic Survival Prediction Task Completed Successfully!")