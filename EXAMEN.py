#JUAN CARLOS DAMIAN DIAZ Y SERGIO JOSEPH CALDERON ADAME
#1._Análisis exploratorio de datos:



import pandas as pd

# Importar los datos desde GitHub
url = "https://raw.githubusercontent.com/gordito100/ArchivosCSVV/main/diabetess.csv"
data = pd.read_csv(url)

# Verificar los primeros registros del dataset
print(data.head())

# Verificar valores faltantes en el dataset
print(data.isnull().sum())

# Tratar los valores faltantes
data = data.dropna()

# Analizar la distribución de las clases "Outcome"
class_distribution = data["Outcome"].value_counts()



#2._Preprocesamiento de datos:




from sklearn.preprocessing import StandardScaler

# Escalado de características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop("Outcome", axis=1))
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])

# Codificación de variables categóricas (si existen)
# Si hay variables categóricas en tu conjunto de datos, debes realizar la codificación correspondiente aquí.

# Verificar el conjunto de datos preprocesado
print(scaled_data.head())
print(data.head())


#3._Entrenamiento y guardado del modelo:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle

# Dividir el conjunto de datos preprocesado en conjuntos de entrenamiento y prueba
X = scaled_data
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Entrenar el modelo SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Evaluación del rendimiento utilizando validación cruzada
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)


# Ajustar los hiperparámetros del modelo SVM (solo como ejemplo, ajusta los parámetros según tus necesidades)
svm_model_tuned = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_model_tuned.fit(X_train, y_train)

# Guardar el modelo entrenado
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model_tuned, f)









