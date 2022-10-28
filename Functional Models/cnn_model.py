import tensorflow as tf
from keras.layers import Layer
from keras.layers import Conv2D,Flatten,Dropout,Dense,MaxPooling2D
from keras.models import Model
from keras import layers, Input
from keras.optimizers import Adam 
from keras.losses import categorical_crossentropy

batch_size = 32
image_size = 256
channels = 3
epochs = 25

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Resizing(image_size, image_size),
  tf.keras.layers.Rescaling(1./255)
])
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])        

# Model Architecture
def model_architecture():
    input_size=(image_size,image_size,channels)
    inputs = Input(shape=input_size)
    
    Convolution_1 = Conv2D(filters=16,kernel_size=(3,3),activation="relu",input_shape=input_size)(inputs)
    Max_1 = MaxPooling2D((2,2))(Convolution_1)
    
    Convolution_2 = Conv2D(filters=32,kernel_size=(3,3),activation="relu")(Max_1)
    Max_2 = MaxPooling2D((2,2))(Convolution_2)
    
    Convolution_3 = Conv2D(filters=32,kernel_size=(3,3),activation="relu")(Max_2)
    Max_3 = MaxPooling2D((2,2))(Convolution_3)
    
    Convolution_4 = Conv2D(filters=64,kernel_size=(3,3),activation="relu")(Max_3)
    Max_4 = MaxPooling2D((2,2))(Convolution_4)
    
    Convolution_5 = Conv2D(filters=128,kernel_size=(3,3),activation="relu")(Max_4)
    Max_5 = MaxPooling2D((2,2))(Convolution_5)
    
    Flatten_layer = Flatten()(Max_5)
    
    Dense_1 = Dense(64,activation="relu")(Flatten_layer)
    
    outputs = Dense(9,activation="relu")(Dense_1)

    cnn_model = Model(inputs=inputs,outputs=outputs)
    
    summary = cnn_model.summary()
    return cnn_model , summary

# Build Model
def compile_model(model,loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']):
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),optimizer='adam',metrics=['accuracy'])

# Train Model
def fit_model(model,train_ds,val_ds,epochs=20):
    history=model.fit(train_ds,epochs=epochs,batch_size=batch_size,verbose=1,validation_data=val_ds)
    return history

#Evaluate Model
def eval_model(model,x_test):
    score = model.evaluate(x_test,verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1])) 
