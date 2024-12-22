# www.codeandcortex.fr
# pip install tensorflow matplotlib numpy scipy pandas scikit-learn

import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Définir les chemins pour sauvegarder les fichiers
model_save_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/mnist_DNN_model.h5'
metrics_save_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/training_metrics.txt'
accuracy_plot_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/accuracy_plot.png'
loss_plot_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/loss_plot.png'
train_images_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/train_sample_images.png'
test_images_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/test_sample_images.png'
errors_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/error_samples.png'
confusion_matrix_path = '/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MNIST/confusion_matrix.png'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Télécharger et charger le dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Affichage de la répartition des données
print("x_train : ", x_train.shape)
print("y_train : ", y_train.shape)
print("x_test  : ", x_test.shape)
print("y_test  : ", y_test.shape)

# Sauvegarde de 5 images aléatoires en fichiers individuels
random_indices = random.sample(range(x_train.shape[0]), 5)
for i, idx in enumerate(random_indices):
    image_path = f"/Users/stephanemeurisse/Documents/Recherche/DNN-dataset-MINSET/sample_image_{i + 1}.png"
    img = Image.fromarray(255 - x_train[idx])  # Inverser les couleurs pour fond blanc
    img.save(image_path)
    print(f"Image sauvegardée sous : {image_path}")

# Affichage de 10 images aléatoires d'entraînement avec leurs labels
train_images_2_show = []
train_titles_2_show = []
for i in range(10):
    r = random.randint(0, x_train.shape[0] - 1)
    train_images_2_show.append(255 - x_train[r])  # Inverser les couleurs pour fond blanc
    train_titles_2_show.append(f"Training image [{r}] = {y_train[r]}")

plt.figure(figsize=(15, 8))
for i, (image, title) in enumerate(zip(train_images_2_show, train_titles_2_show)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.savefig(train_images_path)
print(f"Images d'entraînement sauvegardées sous : {train_images_path}")

# Affichage de 10 images aléatoires de test avec leurs labels
test_images_2_show = []
test_titles_2_show = []
for i in range(10):
    r = random.randint(0, x_test.shape[0] - 1)
    test_images_2_show.append(255 - x_test[r])  # Inverser les couleurs pour fond blanc
    test_titles_2_show.append(f"Test image [{r}] = {y_test[r]}")

plt.figure(figsize=(15, 8))
for i, (image, title) in enumerate(zip(test_images_2_show, test_titles_2_show)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.savefig(test_images_path)
print(f"Images de test sauvegardées sous : {test_images_path}")

# Normalisation et mise en forme des données
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Conversion des labels en one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Construction du modèle DNN
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    # layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Évaluation du modèle
loss, accuracy, = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Sauvegarde des métriques dans un fichier texte
with open(metrics_save_path, 'w') as f:
    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")

# Visualisation et sauvegarde des courbes de performance
plt.figure(figsize=(12, 4))

# Courbe d'Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig(accuracy_plot_path)

# Courbe de Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig(loss_plot_path)

print(f"Courbes sauvegardées sous '{accuracy_plot_path}' et '{loss_plot_path}'")

# Sauvegarde du modèle
model.save(model_save_path)
print(f"Modèle sauvegardé sous '{model_save_path}'")

# Prédictions sur un jeu de test
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# Identification des erreurs
errors = [i for i in range(len(x_test)) if y_pred[i] != y_true[i]]
errors = errors[:min(24, len(errors))]

# Affichage des erreurs avec les prédictions en couleur
plt.figure(figsize=(12, 8))
for i, idx in enumerate(errors[:15]):
    plt.subplot(3, 5, i + 1)
    plt.imshow(255 - x_test[idx].reshape(28, 28), cmap='gray')  # Inverser les couleurs pour fond blanc
    plt.title(f"True: {y_true[idx]} Predict: {y_pred[idx]}", color='red')
    plt.axis('off')
plt.tight_layout()
plt.savefig(errors_path)
print(f"Images d'erreurs sauvegardées sous : {errors_path}")

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap='viridis', xticks_rotation='vertical', values_format=".2f")
plt.title('Matrice de confusion')
plt.savefig(confusion_matrix_path)
plt.show()
print(f"Matrice de confusion sauvegardée sous : {confusion_matrix_path}")

print("Exécution terminée.")

