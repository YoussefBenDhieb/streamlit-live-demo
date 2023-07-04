import tensorflow as tf
import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5"
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5")
])
image_path = "dog_image.jpg"
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
input_array = tf.keras.preprocessing.image.img_to_array(image)
input_array /= 255.0
input_array = tf.expand_dims(input_array, 0)

predictions = model.predict(input_array)
probabilities = tf.nn.softmax(predictions).numpy()

top_1_class_index = tf.argmax(predictions[0]).numpy()
print(top_1_class_index)

labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

# download labels and creates a maps
downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)

classes = []

with open(downloaded_file) as f:
    labels = f.readlines()
    classes = [l.strip() for l in labels]

print(classes[top_1_class_index])