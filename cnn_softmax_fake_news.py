import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


# Load ARFF dataset manually
texts = []
labels = []
with open("fake_news.arff", "r", encoding="utf-8") as f:
    data_started = False
    for line in f:
        if line.strip().upper() == "@DATA":
            data_started = True
            continue
        if data_started and line.strip():
            stripped = line.strip()
            if stripped.endswith("fake"):
                labels.append("fake")
                content = stripped[:-5]
            elif stripped.endswith("real"):
                labels.append("real")
                content = stripped[:-5]
            else:
                continue
            if content.startswith("'"):
                end_idx = content.find("','")
                if end_idx != -1:
                    text = content[1:end_idx]
                else:
                    text = content[1:]
            else:
                text = content
            texts.append(text)

data = pd.DataFrame({'news_text': texts, 'class_label': labels})

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["news_text"] = data["news_text"].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["class_label"])  # fake/real -> 0/1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data["news_text"],
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Tokenization
max_words = 20000
max_len = 300

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

# CNN + Softmax model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    X_train_pad,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
y_pred_prob = model.predict(X_test_pad)
y_pred = y_pred_prob.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("CNN + Softmax Accuracy:", accuracy * 100)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("CNN + Softmax Confusion Matrix:")
print(cm)

# Visualize and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
plt.title("CNN + Softmax Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cnn_softmax_confusion_matrix.png", dpi=300)
plt.show()

print("CNN + Softmax confusion matrix figure saved successfully!")

# Save model
model.save("cnn_softmax_fake_news_model.h5")
print("CNN + Softmax model saved successfully!")
