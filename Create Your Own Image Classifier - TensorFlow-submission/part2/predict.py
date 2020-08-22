#import stuffs
import argparse #command line convert
import tensorflow as tf
import numpy as np
import json
import tensorflow_hub as hub
from PIL import Image

image_size = 224

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image /= 255
    image = image.numpy()
    return image

def predict(image_path, model, top_k):
    im = np.asarray(Image.open(image_path))
    im = np.expand_dims(process_image(im), axis=0)
    prediction = model.predict(im).squeeze()
    classes = prediction.squeeze().argsort()[-top_k:]
    probs = np.empty((0,1), float)
    for index in classes:
        probs = np.append(probs, prediction[index])
    classes += 1
    return probs, classes

parser = argparse.ArgumentParser(description = 'Predict image label using MobileNet deep learning')
parser.add_argument('image_path', action = "store", help = 'the path to the image file')
parser.add_argument('model_name', action = "store", help = 'path to model file .h5')
parser.add_argument('--top_k', action = "store", type = int, dest = "top_k", default = 1, help = 'return TOP_K number of the most likely results (default : 1)')
parser.add_argument('--category_name', action = "store", dest = "category_name", default = 'label_map.json', help = 'load the JSON file that maps the class values to category names (default: label_map.json)')
result = parser.parse_args()


image_path, model_name, top_k, category_name = result.image_path, result.model_name, result.top_k, result.category_name

#load model from model name
model_path = "./" + model_name
loaded_model = tf.keras.models.load_model(model_path, custom_objects = {'KerasLayer':hub.KerasLayer})

#load json file
with open(category_name, 'r') as f:
    class_names = json.load(f)    

#predict
probs, classes = predict(image_path, loaded_model, top_k)
label_names = np.empty((0,1), str)
for index in classes:
    label_names = np.append(label_names, class_names[str(index)])
output = dict(zip(label_names, probs))
print(output)



