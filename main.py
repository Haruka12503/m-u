import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, LSTM, Reshape
from tensorflow.keras.models import Model

# Đường dẫn tới thư mục dữ liệu
base_dir = './dataset_fruit/Fruits Classification'

# Thiết lập các tham số
image_height, image_width = 96, 96
batch_size = 8
num_classes = 5

# Tạo ImageDataGenerator cho huấn luyện với tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo bộ dữ liệu cho huấn luyện, xác thực và kiểm tra
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='sparse'
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(base_dir, 'valid'),
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Kiểm tra số lượng mẫu trong các generator
train_samples = train_generator.samples
valid_samples = valid_generator.samples
test_samples = test_generator.samples

print(f'Training samples: {train_samples}')
print(f'Validation samples: {valid_samples}')
print(f'Test samples: {test_samples}')

# === Mô hình CNN ===
base_model_cnn = MobileNetV2(weights=None, include_top=False, input_shape=(image_height, image_width, 3))
x = base_model_cnn.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions_cnn = Dense(num_classes, activation='softmax')(x)
model_cnn = Model(inputs=base_model_cnn.input, outputs=predictions_cnn)

for layer in base_model_cnn.layers:
    layer.trainable = True

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình CNN
steps_per_epoch = int(np.ceil(train_samples / batch_size))
validation_steps = int(np.ceil(valid_samples / batch_size))

history_cnn = model_cnn.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    epochs=5  # Số lượng epochs
)

# Đánh giá mô hình CNN
cnn_test_loss, cnn_test_accuracy = model_cnn.evaluate(test_generator, steps=test_samples // batch_size)
print(f'CNN Test Loss: {cnn_test_loss:.4f}')
print(f'CNN Test Accuracy: {cnn_test_accuracy * 100:.2f}%')

# === Mô hình CNN + LSTM ===
base_model_lstm = MobileNetV2(weights=None, include_top=False, input_shape=(image_height, image_width, 3))
x = base_model_lstm.output
x = GlobalAveragePooling2D()(x)
x = Reshape((1, -1))(x)  # Định hình lại đầu vào cho lớp LSTM
x = LSTM(128)(x)
x = Dense(128, activation='relu')(x)
predictions_lstm = Dense(num_classes, activation='softmax')(x)
model_lstm = Model(inputs=base_model_lstm.input, outputs=predictions_lstm)

for layer in base_model_lstm.layers:
    layer.trainable = True

model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình CNN + LSTM
history_lstm = model_lstm.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    epochs=5
)

# Đánh giá mô hình CNN + LSTM
lstm_test_loss, lstm_test_accuracy = model_lstm.evaluate(test_generator, steps=test_samples // batch_size)
print(f'CNN + LSTM Test Loss: {lstm_test_loss:.4f}')
print(f'CNN + LSTM Test Accuracy: {lstm_test_accuracy * 100:.2f}%')

# === So sánh Kết quả ===
def plot_comparison(history_cnn, history_lstm):
    plt.figure(figsize=(12, 4))

    # Độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(history_cnn.history['accuracy'], label='CNN Training Accuracy')
    plt.plot(history_cnn.history['val_accuracy'], label='CNN Validation Accuracy')
    plt.plot(history_lstm.history['accuracy'], label='CNN + LSTM Training Accuracy')
    plt.plot(history_lstm.history['val_accuracy'], label='CNN + LSTM Validation Accuracy')
    plt.title('Comparison of Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Tổn thất
    plt.subplot(1, 2, 2)
    plt.plot(history_cnn.history['loss'], label='CNN Training Loss')
    plt.plot(history_cnn.history['val_loss'], label='CNN Validation Loss')
    plt.plot(history_lstm.history['loss'], label='CNN + LSTM Training Loss')
    plt.plot(history_lstm.history['val_loss'], label='CNN + LSTM Validation Loss')
    plt.title('Comparison of Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Vẽ đồ thị so sánh
plot_comparison(history_cnn, history_lstm)
