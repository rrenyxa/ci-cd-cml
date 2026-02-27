import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Генерируем синтетический датасет
X, y = make_classification(1000, random_state=42)

# 2. Разделяем на train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

# 3. Создаём и обучаем модель
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
with open('metrics.txt', 'w') as f:
    f.write('Accuracy: ' + str(accuracy) + '\n')

disp = ConfusionMatrixDisplay.from_estimator(model,
                                             X_test,
                                             y_test,
                                             normalize='true')
plt.savefig("confusion_matrix.png")