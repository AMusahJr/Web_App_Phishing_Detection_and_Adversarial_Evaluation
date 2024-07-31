import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import joblib
import pickle

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('/Users/musahamidujunior/Downloads/Phishing_Email.csv')

# Verify columns in dataset
print("Columns in dataset:", df.columns)

# Use correct column names
text_column = 'Email Text'
label_column = 'Email Type'

# Handle NaN values
df[text_column] = df[text_column].fillna('')

# Convert labels to integers
label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
df[label_column] = df[label_column].map(label_mapping)

# Preprocess data
print("Preprocessing data...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df[text_column])
y = df[label_column]

# Save vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Train Naive Bayes model without scaling or PCA
print("Training Naive Bayes model...")
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X, y, test_size=0.2, random_state=42)
nb_model = MultinomialNB()

# Define hyperparameters for tuning
param_grid_nb = {'alpha': [0.1, 0.5, 1.0]}

# Use GridSearchCV to find the best hyperparameters
grid_search_nb = GridSearchCV(nb_model, param_grid_nb, cv=3)
grid_search_nb.fit(X_train_nb, y_train_nb)

# Get the best Naive Bayes model
best_nb_model = grid_search_nb.best_estimator_
joblib.dump(best_nb_model, 'naive_bayes_tuned.pkl')

# Generate adversarial examples for Naive Bayes
def generate_adversarial_examples_nb(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)
    X_adv = X + noise
    return np.clip(X_adv, 0, 1)

adversarial_X_train_nb = generate_adversarial_examples_nb(X_train_nb.toarray())
adversarial_y_train_nb = y_train_nb

# Combine original and adversarial training data
X_train_combined_nb = np.vstack((X_train_nb.toarray(), adversarial_X_train_nb))
y_train_combined_nb = np.hstack((y_train_nb, adversarial_y_train_nb))

# Retrain the Naive Bayes model with adversarial examples
best_nb_model.fit(X_train_combined_nb, y_train_combined_nb)
joblib.dump(best_nb_model, 'naive_bayes_adversarial_trained.pkl')

# Preprocess data for other models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.toarray())

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Apply PCA
pca = PCA(n_components=100)  # Use fixed number of components
X_pca = pca.fit_transform(X_scaled)

# Save PCA
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Decision Tree model
print("Training Decision Tree model...")
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'decision_tree.pkl')

# Train SVM model
print("Training SVM model...")
svm_model = SVC(max_iter=5000)

# Define hyperparameters for tuning
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Use GridSearchCV to find the best hyperparameters
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=3)
grid_search_svm.fit(X_train, y_train)

# Get the best SVM model
best_svm_model = grid_search_svm.best_estimator_
joblib.dump(best_svm_model, 'svm_tuned.pkl')

# Generate adversarial examples for SVM
adversarial_X_train_svm = generate_adversarial_examples_nb(X_train)  # Using the same function for simplicity
adversarial_y_train_svm = y_train

# Combine original and adversarial training data
X_train_combined_svm = np.vstack((X_train, adversarial_X_train_svm))
y_train_combined_svm = np.hstack((y_train, adversarial_y_train_svm))

# Retrain the SVM model with adversarial examples
best_svm_model.fit(X_train_combined_svm, y_train_combined_svm)
joblib.dump(best_svm_model, 'svm_adversarial_trained.pkl')

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'random_forest.pkl')

# Preprocess data for LSTM
print("Preprocessing data for LSTM...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df[text_column])
X_sequences = tokenizer.texts_to_sequences(df[text_column])
X_padded = pad_sequences(X_sequences, maxlen=500)
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Convert labels to integers
y_train_seq = y_train_seq.astype(int)
y_test_seq = y_test_seq.astype(int)

# Train LSTM model
print("Training LSTM model...")
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
lstm_model.add(LSTM(128, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(128))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.2)
lstm_model.save('lstm_model.h5')

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Training complete and models saved.")
