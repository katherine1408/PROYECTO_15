# 🎥 Sprint 15 – Clasificación Automática de Reseñas Negativas (Film Junky Union)

## 📌 Descripción del Proyecto

**Film Junky Union**, una comunidad para amantes del cine clásico, está desarrollando un sistema automático para **detectar críticas negativas** en reseñas de películas. 

El objetivo de este proyecto es construir y comparar modelos de **aprendizaje automático** que clasifiquen reseñas como **positivas o negativas**, usando datos reales de IMDB. Para considerarse exitoso, el modelo debe alcanzar una métrica **F1 ≥ 0.85** en el conjunto de prueba.

## 🎯 Objetivos del Proyecto

- Explorar los datos y verificar el equilibrio de clases.
- Preprocesar las reseñas de texto.
- Vectorizar las reseñas para entrenar modelos.
- Comparar al menos tres modelos de clasificación:
  - `LogisticRegression`
  - `RandomForestClassifier` o `GradientBoosting`
  - (opcional) `BERT` aplicado a subconjuntos
- Evaluar los modelos con métricas (`F1`, `accuracy`, `confusion matrix`)
- Clasificar reseñas nuevas manuales con los modelos entrenados.

## 📁 Dataset utilizado

- `imdb_reviews.tsv`

Columnas:

- `review`: texto de la reseña
- `pos`: etiqueta de sentimiento (1 = positiva, 0 = negativa)
- `ds_part`: conjunto de pertenencia (`train` o `test`)

## 🧰 Funcionalidades del Proyecto

### 🔍 Análisis Exploratorio

- Cálculo de proporciones de clases
- Visualización de distribución de longitud de reseñas

### 🧹 Preprocesamiento y vectorización

- Limpieza básica del texto: minúsculas, signos, stopwords
- Tokenización y vectorización con:
  - `CountVectorizer`
  - `TfidfVectorizer`

### 🤖 Modelado

- Entrenamiento con múltiples algoritmos de clasificación
- Ajuste básico de hiperparámetros
- Evaluación con función `evaluate_model()`

### ✨ Clasificación personalizada

- Predicción de reseñas escritas manualmente
- Comparación de decisiones entre modelos

### (Opcional) 🧠 BERT
- Vectorización con `BERT_text_to_embeddings()` para un subconjunto reducido
- Evaluación del impacto en rendimiento

## 📊 Herramientas utilizadas

- Python  
- pandas / numpy  
- scikit-learn  
- matplotlib / seaborn  
- nltk / re  
- (opcional) `transformers` de Hugging Face para BERT

---

📌 Proyecto desarrollado como parte del Sprint 15 del programa de Ciencia de Datos en **TripleTen**.
