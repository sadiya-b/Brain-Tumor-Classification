from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from tensorflow.keras.utils import load_img

import json
import numpy as np
from PIL import Image
from pathlib import Path
import os


#import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder 

#Data Processing


# Create your views here.
def index(request):
  context = {'a':1}
  return render(request, 'index.html',context)

def names(number):
  if number==0:
    return 'Its a Tumor'
  else:
    return 'No, Its not a tumor'

def predictImage(request):
  print(os.getcwd())
  print(request)
  print(request.FILES)

  fileObj = request.FILES['filePath']

  fs = FileSystemStorage()

  filePathName = fs.save(fileObj.name,fileObj)
  filePathName = fs.url(filePathName)
    
  uploadedImage = '.'+filePathName
  print(uploadedImage)

  img = load_img(uploadedImage)
  x = np.array(img.resize((128,128)))
  x = x.reshape(1,128,128,3)
  print("done reshaping")


  from tensorflow.keras.models import model_from_json
  json_file = open("model.json", "r")
  loaded_model_json = json_file.read()
  json_file.close()

  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model.h5")
  print("Loaded model from disk")

  #model = load_model("model.h5")
  res = loaded_model.predict(x)
  classification = np.where(res == np.amax(res))[1][0]
  predictedLabel = str(res[0][classification]*100) + '% Confidence This Is ' + names(classification)
  print(predictedLabel)
  print(str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))

  context = {'filePathName':filePathName, 'predictedLabel':predictedLabel}
  return render(request, 'index.html',context)

def daniel(request):
  context = {'a':1}
  return render(request,'daniel.html',context)