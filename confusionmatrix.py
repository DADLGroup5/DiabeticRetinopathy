from sklearn.metrics import classification_report
#Confusion matrix part
y_pred = model.predict(X_test) 
#print(y_pred)
y_pred = model.predict_classes(X_test) #gives the most likely class as output
#print(y_pred)

target_category = ['mild','severe']
#precision, recall, f1score
print(classification_report(y_test,y_pred,target_names=target_category)) 
print(confusion_matrix(y_test,y_pred))