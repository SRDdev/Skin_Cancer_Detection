import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

batch_size = 32
image_size = 256
channels = 3
epochs = 25

#Intake Images from Directory 
def Directory_to_Image(path):
    dataset = tf.keras.utils.image_dataset_from_directory(directory=path,batch_size=batch_size,image_size=(image_size,image_size))
    class_names=dataset.class_names
    out_params = len(np.unique(class_names))
    return dataset , class_names

#Split the data
def  dataset_split(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle_size=10000):
    ds_size=len(ds)
    if shuffle_size:
        ds=ds.shuffle(shuffle_size,seed=42)
        
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    
    train_ds=ds.take(train_size)
    validation_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    
    return train_ds ,validation_ds ,test_ds


#Plot random images from dataset
def plot_random(dataset,class_names):
    plt.figure(figsize=(10,10))
    for image_batch,label_batch in dataset.take(1):
        for i in range(12):
            ax=plt.subplot(3,4,i+1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.axis("off")
            plt.title(class_names[label_batch[i]])

#Remove cache & shuffle
def remove_cache(train_ds,validation_ds,test_ds):
    train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds=validation_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)   


#Resize & Rescale & Data augmentation
def resize_and_rescale(image_size):
    resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(image_size, image_size),
    layers.experimental.preprocessing.Rescaling(1./255)])
def Augmentation():
    data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),])

# Accuracy Plot
def Accuracy_plot(history):
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    val_loss=history.history['val_loss']
    loss=history.history['loss']
    epoch_count = range(1, len(loss) + 1)
    # Visualize loss history
    plt.plot(epoch_count, loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();
    return acc,val_acc,val_loss,loss


# Prediction Image Plot
def Predcition(model,test_ds,class_names,image_numbers):
    for images_batch,label_batch in test_ds.take(image_numbers):
        first_image= images_batch[0].numpy().astype('uint8')
        first_label= label_batch[0].numpy()
        
        print("First image to predict")
        plt.imshow(first_image)
        print("first image's actual label : ",class_names[first_label])
        
        batch_predict=model.predict(images_batch)
        print("first image's predicted label : ",class_names[np.argmax(batch_predict[0])])
        
def save(model,verion=0):
    model.save(f'{model}_v{version}.h5')
    model.save_weights(f'{model}_v{version}_weights.h5')
    print("Model & Weights Saved !!")