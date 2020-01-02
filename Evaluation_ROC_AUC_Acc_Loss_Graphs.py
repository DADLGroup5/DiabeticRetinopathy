# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:14:40 2020

@author: Karthikeyan S
"""


#ROC Curve code
y_pred = model.predict(X_test)

#Keeping only positive outcome probabilities
y_pred = y_pred[:, 1]

ns_fpr , ns_tpr , thresholds = roc_curve(y_test , ns_probs)
fpr , tpr , thresholds = roc_curve(y_test , y_pred)

plt.plot(ns_fpr , ns_tpr , linestyle='--' , label = 'No Skill')
plt.plot(fpr , tpr , marker = '.' , label = 'Built model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('ROC.png')
plt.show()

#AUC Score
lr_auc = roc_auc_score(y_test, y_pred)
print('AUC Score : %.3f' % (lr_auc))

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

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

#Training loss vs Validation loss
plt.plot(training_loss)
plt.plot(validation_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('LossVS.png')
plt.show()
