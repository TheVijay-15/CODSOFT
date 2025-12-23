import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load 
df = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/CODSOFT/Iris_Flower_Classification_Task3/IRIS.csv")

print(" Dataset Head ")
print(df.head())

# 2. Preprocessing
# Splitting x & y
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\n Classification Report")
print(classification_report(y_test, y_pred))

# 5. output
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Iris Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('output_result.png') 
print("\nOutput image 'output_result.png' has been generated.")