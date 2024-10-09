import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Mengatur parameter
img_height, img_width = 150, 150
batch_size = 1  # Menggunakan batch_size 1 karena hanya ada satu gambar per folder

# Menggunakan ImageDataGenerator untuk mempersiapkan data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train',  # Pastikan path ini sesuai dengan lokasi folder data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Membangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 kelas untuk 4 ekspresi
])

# Mengkompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(train_generator, epochs=10)  # Ganti 10 dengan jumlah epoch yang diinginkan

# Mulai menangkap video dari kamera
cap = cv2.VideoCapture(0)  # 0 untuk kamera default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mengubah ukuran dan mengolah gambar
    img = cv2.resize(frame, (img_width, img_height))
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch

    # Membuat prediksi
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])  # Mengambil kelas dengan probabilitas tertinggi

    # Mendefinisikan label untuk ekspresi
    classes = ['Marah', 'Bingung', 'Sedih', 'Senang']  # Ganti dengan nama kelas sesuai folder
    predicted_class = classes[class_index]

    # Menampilkan prediksi di frame
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Menampilkan frame
    cv2.imshow('Kamera - Pengenalan Ekspresi Wajah', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup semua jendela
cap.release()
cv2.destroyAllWindows()
