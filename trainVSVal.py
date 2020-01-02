# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:08:41 2020

@author: Karthikeyan S
"""

#Training accuracy vs Validation accuracy
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

plt.plot(training_accuracy)
plt.plot(validation_accuracy)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('AccuracyVS.png')
plt.show()

#Training loss vs Validation loss
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

plt.plot(training_loss)
plt.plot(validation_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('LossVS.png')
plt.show()
