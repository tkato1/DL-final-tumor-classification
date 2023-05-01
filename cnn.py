import tensorflow as tf
import numpy as np

from preprocess import load_data

from sklearn.model_selection import train_test_split


X, y = load_data("data/test", "jpegs/uncropped32/", "labels/labels.txt", downsampling_factor=16)
X = tf.convert_to_tensor(X, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, 3, dtype=tf.float32)
y = tf.reshape(y, (y.shape[0], y.shape[2]))
print(X.shape, y.shape)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = tf.convert_to_tensor(X_train)
# X_test = tf.convert_to_tensor(X_test)
# y_train = tf.convert_to_tensor(y_train)
# y_test = tf.convert_to_tensor(y_test)

# print(f"X_train: {X_train.shape} y_train: {y_train.shape}")
# print(f"X_test: {X_test.shape} y_test: {y_test.shape}")

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(
  loss=tf.keras.losses.categorical_crossentropy,
  optimizer=tf.keras.optimizers.Adam(),
  metrics=['accuracy']
)

model.build(X.shape)
model.summary()

print(y[-1:])

# Train the model
model.fit(
  X[:8],
  y[:8],
  epochs=10,
  batch_size=32,
  validation_data = (y[8:], y[8:])
)

# # Evaluate the model
# model.evaluate(
#   X_test,
#   y_test
# )