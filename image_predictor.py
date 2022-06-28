import streamlit as st
from PIL import Image
from keras.models import load_model

st.header("Image Predictor")


html_temp = """
<div style="background-color:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;background-color: Blue;">Try again and again!!!</h2>
<h3 style="color:red;text-align:center;">You will crack it Fahim</h3>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)


# load model
Fundus_covid19 = load_model("DR_VGG19_new.h5")

uploaded_file = st.file_uploader("Choose a file")

im = Image.open(uploaded_file)
im = im.resize((224,224))
im = np.array(im)
im = im/255
im = np.expand_dims(im,axis=0)

result = Fundus_covid19.predict(im)

if result[0][0] > result[0][1]:
    print("Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][0]*100)))
else:
  print("NO Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][1])*100))



