import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5"
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5")
])

st.title("AI Image Classifier")
st.write("MAde with Streamlit")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg"])

if uploaded_image is not None:
    with st.spinner('Waiting for the prediction'):
        image = tf.keras.preprocessing.image.load_img(uploaded_image, target_size=(224, 224))
        input_array = tf.keras.preprocessing.image.img_to_array(image)
        input_array /= 255.0
        input_array = tf.expand_dims(input_array, 0)

        predictions = model.predict(input_array)
        probabilities = tf.nn.softmax(predictions).numpy()

        top_1_class_index = tf.argmax(predictions[0]).numpy()

        labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

        # download labels and creates a maps
        downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)

        classes = []

        with open(downloaded_file) as f:
            labels = f.readlines()
            classes = [l.strip() for l in labels]

        st.success(f"This is an image of a {classes[top_1_class_index]}")
        st.image(uploaded_image)
        with st.expander("Click her to see more statistics"):
            st.write("more statistics")