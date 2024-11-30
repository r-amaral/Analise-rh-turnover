import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('./table/Dados_RH_Turnover.csv', delimiter=';')

df = pd.get_dummies(df, columns=['DeptoAtuacao', 'Salario'], drop_first=True)

y = df['SaiuDaEmpresa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Neural Network': MLPClassifier(max_iter=300)
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'Accuracy': accuracy,
        'Confusion Matrix': matrix
    }

for name, metrics in results.items():
    accuracy = metrics['Accuracy']
    confusion = metrics['Confusion Matrix']
    
    print(f"Modelo: {name}")
    print(f"- Acurácia: {accuracy:.2%}")
    print(f"- Matriz de Confusão:\n{confusion}\n")
    
    tn, fp, fn, tp = confusion.ravel()
    print(f"Detalhes:")
    print(f"  Verdadeiros Negativos (TN): {tn}")
    print(f"  Falsos Positivos (FP): {fp}")
    print(f"  Falsos Negativos (FN): {fn}")
    print(f"  Verdadeiros Positivos (TP): {tp}\n")

    print(f"Resumo do desempenho do modelo '{name}':")
    if accuracy >= 0.9:
        print(f"  Alta precisão. Modelo bem adequado para prever a saída de funcionários.")
    elif 0.75 <= accuracy < 0.9:
        print(f"  Desempenho moderado. Pode ser útil, mas precisa de ajustes.")
    else:
        print(f"  Baixa precisão. Não é confiável para previsões robustas.\n")
