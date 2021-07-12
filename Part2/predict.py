
from PIL import Image
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import argparse

def process_image(image):
    image_tensor = tf.convert_to_tensor(image)
    
    image_tensor= tf.cast(image_tensor,tf.float32)
    image_tensor = tf.image.resize(image_tensor, (224, 224))
    image_tensor = image_tensor /255
    
    image_numpy = image_tensor.numpy()
   # print(image_tensor.shape)  # must return NumPy array with shape (224, 224, 3).
    return image_numpy


def predict(image_path, model_name, top_k, category_names):
    
    #1.process image
    im = Image.open(image_path)
    image_numpy = np.asarray(im)
    processed_image = process_image(image_numpy)
    processed_image_expand = np.expand_dims(processed_image, axis=0)
    
    #2.process model
    
    #2.1 load the model
    reloaded_keras_model = tf.keras.models.load_model(
    './{}'.format(model_name),custom_objects={'KerasLayer': hub.KerasLayer})
    
    #2.2 make prediction
    ps = reloaded_keras_model.predict(processed_image_expand) #return numpy array 1X102
    

    #3. take the top 5 prediction
    classes_ = list( (-ps[0,]).argsort()[:top_k] )
    
    probs_list=[]
    for x in classes_:
        probs_list.append(ps[0,x])
    
    classes_list = [] # convert to str
    for item in classes_:
        classes_list.append(str(item+1))
    
    #4. take the classes name
    
    #4.1 download josn
    with open('{}'.format(category_names), 'r') as f:
        class_names = json.load(f)
    #4.2
    class_names_list = []
                    
    for x in classes_list:
        class_names_list.append(class_names[x])
        


    
    return probs_list,class_names_list
    

  
# 1.initialize
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# 2.add the parameters 

#2.1 assential
parser.add_argument('image_path',type=str, default=None, help='Path to input image')
parser.add_argument('saved_model',type=str, default=None, help='Load saved model for prediction')

#2.2 optional
parser.add_argument('--top_k', type=int, default=5, help='Predict the top K character probabilities')
parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to JSON file mapping categories to names')

# 3. get the argument
args = parser.parse_args()

args.image_path
args.saved_model
args.top_k
args.category_names

probs, classes_name = predict(args.image_path, args.saved_model,args.top_k,args.category_names)
for x in range(args.top_k):
    print("Image name:",classes_name[x] , "\nprobabilty:",probs[x])
    
max_value = max(probs)
max_index = probs.index(max_value)
print('so the top predction image is:',classes_name[max_index])


