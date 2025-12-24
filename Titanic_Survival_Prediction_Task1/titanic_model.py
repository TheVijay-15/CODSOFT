import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/CODSOFT/Titanic_Survival_Prediction_Task1/Titanic-Dataset.csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin', 'PassengerId', 'Ticket'], inplace=True)

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
df.drop(columns=['Name'], inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True)
}

print("MODEL COMPARISON RESULTS")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cv_score = cross_val_score(model, X, y, cv=5).mean()
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  CV Score:  {cv_score:.4f}")

best_model_name = max(models.keys(), key=lambda x: accuracy_score(y_test, models[x].predict(X_test)))
best_model = models[best_model_name]

print(f"SELECTED MODEL: {best_model_name}")
print(f"Reason: Highest accuracy among all models")

y_pred_best = best_model.predict(X_test)
print(f"\nFinal Model Accuracy: {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

plt.figure(figsize=(10,6))
model_names = list(models.keys())
accuracies = [accuracy_score(y_test, models[name].predict(X_test)) for name in model_names]

bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Model Comparison: Accuracy Scores', fontsize=14, fontweight='bold')
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nModel comparison visualization saved as 'model_comparison.png'")

plt.figure(figsize=(8,5))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Ticket Class', fontsize=12, fontweight='bold')
plt.xlabel('Passenger Class', fontsize=11)
plt.ylabel('Survival Rate', fontsize=11)
plt.savefig('survival_by_class.png', dpi=300, bbox_inches='tight')
plt.show()
print("Survival analysis visualization saved as 'survival_by_class.png'")