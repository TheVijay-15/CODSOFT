import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/CODSOFT/Iris_Flower_Classification_Task3/IRIS.csv")

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Species Distribution:\n{df['species'].value_counts()}")

df['species'] = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

plt.figure(figsize=(12, 8))
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle('Iris Dataset Feature Relationships', y=1.02)
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

X = df.drop('species', axis=1)
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Naive Bayes': GaussianNB()
}

param_grids = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']},
    'Decision Tree': {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15, None]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5, 1.0]},
    'Support Vector Machine': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']},
    'Naive Bayes': {}
}

results = []

print("\nmodel training with gridsearchcv (5-Fold CV)")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_score = grid_search.best_score_
    
    results.append({
        'Model': name,
        'Best Parameters': grid_search.best_params_,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Score': cv_score
    })
    
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  CV Accuracy: {cv_score:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)


print("resuts :Sorted by Accuracy")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Score']].round(4).to_string(index=False))

best_model_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_params = results_df.loc[best_model_idx, 'Best Parameters']

print(f"\nchoose best model: {best_model_name}")
print(f"Best Parameters: {best_model_params}")
print(f"Test Accuracy: {results_df.loc[best_model_idx, 'Accuracy']:.4f}")

final_model = GridSearchCV(
    models[best_model_name], 
    param_grids[best_model_name], 
    cv=5, 
    scoring='accuracy'
).fit(X_scaled, y).best_estimator_

y_pred_final = final_model.predict(X_test)

print(f"\nfinal performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Setosa', 'Versicolor', 'Virginica']))

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
model_names = results_df['Model']
accuracies = results_df['Accuracy']
cv_scores = results_df['CV Score']
x = np.arange(len(model_names))
width = 0.35
bars1 = plt.bar(x - width/2, accuracies, width, label='Test Accuracy', color='blue', alpha=0.7)
bars2 = plt.bar(x + width/2, cv_scores, width, label='CV Accuracy', color='green', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison: Test vs Cross-Validation Accuracy')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Setosa', 'Versicolor', 'Virginica'],
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.title('Confusion Matrix - Best Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(2, 2, 3)
feature_names = df.columns.drop('species').tolist()
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
elif hasattr(final_model, 'coef_'):
    coef = final_model.coef_[0]
    plt.bar(range(len(coef)), coef, align='center')
    plt.xticks(range(len(coef)), feature_names, rotation=45)
    plt.title('Feature Coefficients')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')

plt.subplot(2, 2, 4)
species_names = ['Setosa', 'Versicolor', 'Virginica']
species_colors = ['red', 'green', 'blue']
for i in range(3):
    species_data = df[df['species'] == i]
    plt.scatter(species_data['sepal_length'], species_data['petal_length'], 
                label=species_names[i], alpha=0.6, color=species_colors[i])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Sepal vs Petal Length by Species')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_classification_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVISUALIZATIONS SAVED:")
print("1. iris_pairplot.png - Feature relationships")
print("2. iris_classification_results.png - Model comparison and results")

predictions_df = pd.DataFrame({
    'Actual': [['Setosa', 'Versicolor', 'Virginica'][i] for i in y_test],
    'Predicted': [['Setosa', 'Versicolor', 'Virginica'][i] for i in y_pred_final]
})
print("\nSample Predictions:")
print(predictions_df.head(10).to_string(index=False))

print("\nTASK 3 - iris flower classification")