from PIL import Image
import numpy as np
import numpy as np
from tensorflow.keras.models import model_from_json

# open method used to open different extension image file
img_path = 'C:/Users/himan/ML/PycharmProjects/DeepLearning/Yulia_classification/data/sharp/1.JPG'
im = Image.open(img_path) 
newsize = (96, 96)
im = im.resize(newsize)

# resizing features in accordance with CNN
X = np.array(im).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Normalising X and converting labels to categorical features
X = X.astype('float32')
X /= 255

# load json and create model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# using same image for quality,rotate and mode
prediction = loaded_model([X, X, X])

classes = ['quality', 'rotate', 'mode']
for i in range(len(classes)):
    print(f'Prediction for {classes[i]}',np.array(prediction[i]).argmax(axis=1)[0])   