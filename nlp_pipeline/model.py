
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
!pip install datasets
!pip install transformers
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
import json
import nltk
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tag import PerceptronTagger
from nltk.corpus import words
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter
nltk.download('averaged_perceptron_tagger')

def preprocess_text(text):
    # Removing punctuation and symbols
    text = re.sub(r'[^\w\s]', '', text)

    # Lowercasing the text
    text = text.lower()

    # Removing stopwords
    stopwords_set = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stopwords_set])

    # Removing special characters
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)

    # Removing frequent words
    word_count = Counter(text.split())
    frequent_words = set(word for (word, wc) in word_count.most_common(3))
    text = " ".join([word for word in text.split() if word not in frequent_words])

    # Removing rare words
    rare_words = set(word for (word, wc) in word_count.most_common()[:-10:-1])
    text = " ".join([word for word in text.split() if word not in rare_words])

    # Stemming
    stemmer = SnowballStemmer(language='english')
    text = " ".join([stemmer.stem(word) for word in text.split()])

    # Lemmatization
    lem = WordNetLemmatizer()
    text = " ".join([lem.lemmatize(word) for word in text.split()])

    # POS tagging
    tagger = PerceptronTagger()
    text = " ".join([tag for word, tag in tagger.tag(text.split())])

    # Spelling Correction
    text = TextBlob(text).correct()

    return text

# Load and preprocess data
df = pd.read_csv("data.csv", encoding='ISO-8859-1')
df.dropna(inplace=True)
df['selected_text'] = df['selected_text'].apply(preprocess_text)

# Checking first few rows of preprocessed data
print(df.head())


# Usage example:
#train_bert_classifier("data.csv", num_labels=3)

def train_bert_classifier(data_file, num_labels, num_epochs=3, batch_size=32, learning_rate=2e-5):
    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    class BERTForClassification(tf.keras.Model):

      def __init__(self, bert_model, num_labels):
          super().__init__()
          self.bert = bert_model
          self.fc = tf.keras.layers.Dense(num_labels, activation='softmax')

      def call(self, inputs):
          x = self.bert(inputs)[1]
          return self.fc(x)
    model = BERTForClassification(model, num_classes=num_labels)
    # Load and preprocess data
    df = pd.read_csv(data_file, encoding='ISO-8859-1')
    df.dropna(inplace=True)
    df['selected_text'] = df['selected_text'].apply(preprocess_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df['selected_text'], df['sentiment'], test_size=0.2, random_state=42)

    # Convert Pandas Series to list
    X_train = [str(text) for text in X_train]
    X_test = [str(text) for text in X_test]

    # Tokenize list of texts and convert to PyTorch tensors
    encoded_X_train = tokenizer(X_train, padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_X_test = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # Initialize a label encoder
    label_encoder = LabelEncoder()

    # Fit the encoder on the sentiment labels and transform them to numerical values
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Create TensorDataset
    train_dataset = TensorDataset(
        encoded_X_train.input_ids,
        encoded_X_train.attention_mask,
        torch.tensor(y_train, dtype=torch.long)  # Ensure the labels are of type long
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {average_loss:.4f}")

    print("Training completed.")

    # Evaluate
    model.eval()
    y_pred = []

    with torch.no_grad():
        for text in encoded_X_test.input_ids:
            output = model(input_ids=text.unsqueeze(0))
            logits = output.logits
            predicted_class = torch.argmax(logits, dim=1)
            y_pred.append(predicted_class.item())

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

from datasets import load_dataset
emotions = load_dataset('SetFit/emotion')

train_bert_classifier(emotions, 6)