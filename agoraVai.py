import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Carregar os dados (substitua pelo seu dataset)
df = pd.read_csv("meu_arquivo.csv")  # Caso esteja usando um CSV, por exemplo.

# Exemplo de dados
# df = pd.DataFrame({
#     'reviewBody': ['Texto de exemplo 1', 'Texto de exemplo 2', 'Texto de exemplo 3'],
#     'alternativeName': ['fake', 'relevante', 'irrelevante']
# })

# Limpeza de texto: remoção de caracteres especiais, números, e normalização
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Substitui múltiplos espaços por um único
    text = re.sub(r'[^a-záéíóúãõâêîôûàèìòùç]+', ' ', text)  # Remove caracteres especiais e números
    text = text.strip()
    return text

# Aplicando a limpeza no reviewBody
df['cleaned_review'] = df['reviewBody'].apply(clean_text)

# Codificando as classes (por exemplo, fake, relevante, irrelevante)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['alternativeName'])

# Divisão do dataset entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.2, random_state=42)

# Criar o modelo de classificação com Naive Bayes + TF-IDF
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='portuguese', ngram_range=(1, 2), max_features=5000),
    MultinomialNB()
)

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Previsão no conjunto de teste
y_pred = pipeline.predict(X_test)

# Avaliação do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Exemplo de como fazer previsões
def predict_text(texto):
    cleaned_text = clean_text(texto)
    prediction = pipeline.predict([cleaned_text])
    predicted_class = label_encoder.inverse_transform(prediction)
    return predicted_class[0]

# Teste com exemplo
texto_exemplo = "Texto de exemplo para verificar se é fake ou relevante"
predicao = predict_text(texto_exemplo)
print(f"A classificação do texto é: {predicao}")
