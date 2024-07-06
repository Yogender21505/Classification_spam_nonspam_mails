# Spam vs. Non-Spam Email Classification using BERT

This project demonstrates classification of emails as spam or non-spam using BERT (Bidirectional Encoder Representations from Transformers) and TensorFlow. The dataset used is sourced from Kaggle.

## Table of Contents
- [Introduction](#introduction)
- [Tech Stacks and Techniques](#tech-stacks-and-techniques)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [License](#license)

## Introduction

This project uses a pre-trained BERT model to encode email texts into embeddings, and then a simple neural network to classify these embeddings as spam or non-spam. It covers key concepts in Natural Language Processing (NLP) and Information Retrieval.

## Tech Stacks and Techniques

### Tech Stacks
- **Programming Language**: Python
- **Libraries**: TensorFlow, TensorFlow Hub, TensorFlow Text, Pandas, Scikit-learn
- **Tools**: Jupyter Notebook, Google Colab (optional)

### Techniques
- **Natural Language Processing (NLP)**: Text preprocessing, embedding generation using BERT
- **Machine Learning (ML)**: Supervised learning, binary classification
- **Information Retrieval**: Text embedding, cosine similarity calculation

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Yogender21505/Classification_spam_nonspam_mails.git
    cd Classification_spam_nonspam_mails
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Import Libraries and Load Dataset**

    ```python
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text
    import pandas as pd

    df = pd.read_csv("spam.csv")
    df.head(5)
    ```

2. **Preprocess Dataset**

    Convert the `Category` column to a binary `spam` column where spam emails are marked as `1` and non-spam emails as `0`.

    ```python
    df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    df.head()
    ```

3. **Split Dataset**

    Split the data into training and testing sets. Stratification ensures both sets have a balanced proportion of spam and non-spam emails.

    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], stratify=df['spam'])
    ```

4. **Load BERT Model and Get Embeddings**

    Load BERT's preprocessing and encoder layers from TensorFlow Hub and create a function to get sentence embeddings.

    ```python
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    def get_sentence_embedding(sentences):
        preprocessed_text = bert_preprocess(sentences)
        return bert_encoder(preprocessed_text)['pooled_output']
    ```

    You can test the function with some sample sentences:

    ```python
    get_sentence_embedding([
        "500$ discount. hurry up", 
        "Bhavin, are you up for a volleyball game tomorrow?"
    ])
    ```

## Model Training

1. **Build Model**

    Construct a functional model using BERT embeddings. The model includes a dropout layer and a dense layer with a sigmoid activation function for binary classification.

    ```python
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

    model = tf.keras.Model(inputs=[text_input], outputs=[l])
    ```

2. **Compile Model**

    Compile the model with the Adam optimizer and binary cross-entropy loss function. Accuracy is used as the evaluation metric.

    ```python
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    ```

3. **Train Model**

    Train the model for 5 epochs with the training data.

    ```python
    model.fit(X_train, y_train, epochs=5)
    ```

## Evaluation

Evaluate the model's performance on the test set:

```python
model.evaluate(X_test, y_test)
```

## Inference
Use the trained model to predict whether new messages are spam or not:
```python
reviews = [
    'Reply to win Â£100 weekly! Where will the 2006 FIFA World Cup be held? Send STOP to 87239 to end service',
    'You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p pÂ£3.99',
    'it to 80488. Your 500 free text messages are valid until 31 December 2005.',
    'Hey Sam, Are you coming for a cricket game tomorrow',
    "Why don't you wait 'til at least wednesday to see if you get your ."
]

model.predict(reviews)
```
The `model.predict` function outputs the probability of each review being spam, with values closer to 1 indicating a higher likelihood of spam.
