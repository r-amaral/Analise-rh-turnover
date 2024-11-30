import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregar dados
df = pd.read_csv('./table/Dados_RH_Turnover.csv', delimiter=';')

# Transformar variáveis categóricas em variáveis dummy
df = pd.get_dummies(df, columns=['DeptoAtuacao', 'Salario'], drop_first=True)

# Separar variáveis independentes (X) e dependente (y)
X = df.drop(columns='SaiuDaEmpresa')
y = df['SaiuDaEmpresa']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dicionário com os modelos
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Neural Network': MLPClassifier(max_iter=300)
}

# Dicionário para armazenar os resultados
results = {}

# Treinar e avaliar os modelos
for name, model in models.items():
    # Treinar o modelo
    model.fit(X_train, y_train)
    # Fazer previsões
    y_pred = model.predict(X_test)
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    # Armazenar resultados
    results[name] = {
        'Accuracy': accuracy,
        'Confusion Matrix': matrix
    }

# Exibir os resultados
for name, metrics in results.items():
    accuracy = metrics['Accuracy']
    confusion = metrics['Confusion Matrix']
    
    print(f"Modelo: {name}")
    print(f"- Acurácia: {accuracy:.2%}")
    print(f"- Matriz de Confusão:\n{confusion}\n")
    
    # Detalhes da matriz de confusão
    tn, fp, fn, tp = confusion.ravel()
    print(f"Detalhes:")
    print(f"  Verdadeiros Negativos (TN): {tn}")
    print(f"  Falsos Positivos (FP): {fp}")
    print(f"  Falsos Negativos (FN): {fn}")
    print(f"  Verdadeiros Positivos (TP): {tp}\n")

    # Resumo da performance
    print(f"Resumo do desempenho do modelo '{name}':")
    if accuracy >= 0.9:
        print(f"  Alta precisão. Modelo bem adequado para prever a saída de funcionários.")
    elif 0.75 <= accuracy < 0.9:
        print(f"  Desempenho moderado. Pode ser útil, mas precisa de ajustes.")
    else:
        print(f"  Baixa precisão. Não é confiável para previsões robustas.\n")
