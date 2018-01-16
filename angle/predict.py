# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.layers import Input,Dense
from keras.models import Model
import numpy as np
from PIL import Image
##编译模型，以较小的学习参数进行训练
from keras.optimizers import SGD


def load():
    vgg = VGG16(weights=None,input_shape=(224,224,3))
    ##修改输出层 3个输出
    x  = vgg.layers[-2].output
    predictions_class = Dense(4, activation='softmax', name='predictions_class')(x)
    prediction = [predictions_class]
    model = Model(inputs=vgg.input, outputs=prediction)
    sgd = SGD(lr=0.00001, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('./angle/modelAngle.h5')
    return model
model = load()
def predict(path=None,img=None):
    ROTATE = [0,90,180,270]
    if path is not None:
       im = Image.open(path).convert('RGB').resize((224,224))
    elif img is not None:
       im = Image.fromarray(img).convert('RGB').resize((224,224))
    img = np.array(im)
    img = preprocess_input(img.astype(np.float32))
    pred = model.predict(np.array([img]))
    index = np.argmax(pred,axis=1)[0]
    return ROTATE[index]

