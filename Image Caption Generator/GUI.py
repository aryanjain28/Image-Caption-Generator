from keras.models import load_model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from pickle import load
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from tkinter import *
from PIL import ImageTk, Image 
from tkinter import filedialog 
from tkinter import messagebox
import keras

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

fileName = 'xxx'
m = Xception()
imageModel = Model(inputs=m.layers[0].input, outputs=m.layers[-2].output)
mainModel = load_model('./Xception_model_4.h5')

tokenizer = open('tokenizer.pkl', 'rb')
tokenizer = load(tokenizer)

w2i = tokenizer.word_index
i2w = {j:i for i,j in w2i.items()}

root = Tk() 
root.title("Image Loader") 
root.resizable(width = True, height = True) 
root.minsize(1000, 700)

def getCaption(fileName):
    print(fileName)
    WIDTH = 299
    HEIGHT = 299
    image = load_img(fileName, target_size=(WIDTH,HEIGHT))
    image = img_to_array(image)
    image.shape = (1,) + image.shape
    image = preprocess_input(image)
    imageInput = imageModel.predict(image)[0]
    
    maxLen = 34
    sentence = nextWord = 'startsentence'
    
    intWord = w2i[nextWord]
    textInput = pad_sequences([[intWord]], maxlen=maxLen)[0]
    
    for _ in range(maxLen):
        prediction = mainModel.predict([[imageInput], [textInput]])
        nextWord = i2w[np.argmax(prediction)]
        sentence += ' ' + nextWord
        
        if nextWord == 'endsentence':
            break

        textInput = list(textInput)
        textInput.append(np.argmax(prediction))
        textInput.pop(0)
        textInput = np.array(textInput)
        
    sentence = ' '.join(sentence.split(' ')[1:-1]).upper() + '.'
    return sentence
    
def open_img():
    global panel
    global fileName
    
    x = openfilename() 
    
    if x == '':
        fileName = 'xxx'
        messagebox.askokcancel("Error : 404", 'Please upload image for prediction.')
        return
    
    img = Image.open(x) 
    
    W = img.size[0]
    H = img.size[1]
        
    if W > H:
        if W > 600:
            H = (600*H)//W
            W = 600

        if W < 100:
            H = (100*H)//W
            W = 100
    else:
        if H > 500:
            W = (500*W)//H
            H = 500

        if H < 200:
            W = (200*W)//H
            H = 200
    
    img = img.resize((W, H), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img)
    
    try:
        panel.destroy()
    except:
        None
    
    panel = Label(root, image = img) 
    panel.image = img 
    panel.place(relx=0.5, rely=0.5, anchor=CENTER)
    
def openfilename():
    global fileName
    fileName = filedialog.askopenfilename(title ='Browser') 
    return fileName

def imageInfo():
    global fileName
    print(fileName)
    if fileName == 'xxx':
        print('Please upload image')
        messagebox.askokcancel("Error : 404", 'Please upload image')
    else:
        caption = getCaption(fileName)
        messagebox.askokcancel('Prediction', caption)
        print(caption)

btn1 = Button(root, text ='Upload Image', font=('Ariel', 15), command = open_img) 
btn2 = Button(root, text ='Predict', font=('Ariel', 15), command=imageInfo)

btn1.place(relx=0.40, rely=0.02, anchor=N)
btn2.place(relx=0.60, rely=0.02, anchor=N)

root.mainloop()
