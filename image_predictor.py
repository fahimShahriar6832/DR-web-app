import streamlit as st
st.header("Image Predictor")


html_temp = """
<div style="background-color:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;background-color: Blue;">Try again and again!!!</h2>
<h3 style="color:red;text-align:center;">You will crack it Fahim</h3>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

import cv2
import keras
from keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image

def main():
  file_uploaded = st.file_uploader("Upload Query image", type = ['jpg','png','jpeg'])
  if file_uploaded is not None:
    image = Image.open(file_uploaded)
   # plt.imshow(image)
    #plt.axis('off')


def predict_class(image):
  classifier_model = tensorflow.keras.models.load_model('DR_VGG19.h5')#Give the model path
  shape = ((224,224,3))#Give the shape
  model = tensorflow.keras.Sequential(hub[hub.KerasLayer(classifier_model,input_shape = shape)])
  test_image = image.resize((224,224))
  test_image = preprocessing.image.img_to_array(test_image)
  test_image = test_image/255
  test_image = np.expand_dims(test_image, axis = 0)
  result = model.predict(test_image) 

  if result[0][0] > result[0][1]:
    print("Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][0]*100)))
  else:
    print("NO Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][1])*100))


if __name__ == "__main__":
  main()
