import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
print("We're using TF", tf.__version__)

# Загружаем датасет Fashion MNIST
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

# Преобразуем метки в one-hot encoding
y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
y_val_oh = tf.keras.utils.to_categorical(y_val, 10)

# Очищаем граф для повторных запусков
tf.keras.backend.clear_session()

# Создаем модель
model = tf.keras.models.Sequential([    
    tf.keras.layers.Dense(units=128, input_dim=784, activation='elu'),
    tf.keras.layers.Dense(units=128, activation='elu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Компилируем модель
model.compile(
    loss='categorical_crossentropy',  # минимизируем кросс-энтропию
    optimizer='adam',
    metrics=['accuracy']  # выводим процент правильных ответов
)

# Нормализуем входные данные
x_train_float = x_train.astype(np.float32) / 255 - 0.5
x_val_float = x_val.astype(np.float32) / 255 - 0.5

# Обучаем модель
model.fit(
    x_train_float.reshape(-1, 28*28),
    y_train_oh,
    batch_size=64,  # 64 объекта для подсчета градиента на каждом шаге
    epochs=10,  # 10 проходов по датасету
    validation_data=(x_val_float.reshape(-1, 28*28), y_val_oh)
)

(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()[1]

x_test_float = x_test.astype(np.float32) / 255 - 0.5
x_test_reshaped = x_test_float.reshape(-1, 28*28)

predictions = model.predict(x_test_reshaped)
н
predicted_labels = np.argmax(predictions, axis=1)

print("Predicted labels for the first 10 images:", predicted_labels[:10])

print("True labels for the first 10 images:", y_test[:10])
