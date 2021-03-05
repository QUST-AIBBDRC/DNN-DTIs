import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from HH import to_categorical,categorical_probas_to_classes,calculate_performace
from sklearn.preprocessing import scale,StandardScaler
from keras.layers import Dense, merge,Input,Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Conv1D, AveragePooling1D
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

shu1 = pd.read_csv("NR_zheng.csv")
shu2 = pd.read_csv("NR_fu.csv")
shu1 = np.array(shu1)
shu2 = np.array(shu2)
shu = np.concatenate((shu1,shu2),axis=0)
[row1,column1]=np.shape(shu1)
[row2,column2]=np.shape(shu2)
label_P = np.ones(int(row1))
label_N = np.zeros(int(row2))
label = np.hstack((label_P,label_N))
shu=np.array(shu)
label=np.array(label)
shu = scale(shu)

xgb_model=xgb.XGBClassifier()
xgbresult1=xgb_model.fit(shu,label.ravel())
feature_importance=xgbresult1.feature_importances_
feature_number=-feature_importance
H1=np.argsort(feature_number)
mask=H1[:300]
train_data=shu[:,mask]

X=train_data
y=label

X_resampled_smote, y_resampled_smote = SMOTE(ratio={1:17538}).fit_sample(X,y)
X_resampled_smote, y_resampled_smote = RandomUnderSampler(ratio={0:17538}).fit_sample(X_resampled_smote, y_resampled_smote)
sorted(Counter(y_resampled_smote).items())

data= X_resampled_smote
label=y_resampled_smote
#shu=scale(data_AAindex)
#shu=scale(data)
X=data
y=label
[m,n]=np.shape(X)
[sample_num,input_dim]=np.shape(X)
out_dim=2
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5


def get_CNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Conv1D(filters = 10, kernel_size = 10, padding = 'same', activation= 'relu'))
    model.add(AveragePooling1D(pool_size=2,strides=2,padding="SAME"))
    model.add(Conv1D(filters = 10, kernel_size =  10, padding = 'same', activation= 'relu'))
    model.add(AveragePooling1D(pool_size=2,strides=2,padding="SAME"))
    #model.add(MaxPooling1D(pool_size=2,padding="SAME"))  
    model.add(Conv1D(filters = 10, kernel_size =  10, padding = 'same', activation= 'relu'))
    model.add(AveragePooling1D(pool_size=2,strides=2,padding="SAME"))
    model.add(Conv1D(filters = 10, kernel_size =  10, padding = 'same', activation= 'relu'))
    model.add(AveragePooling1D(pool_size=2,strides=2,padding="SAME"))
    #model.add(MaxPooling1D(pool_size=2,padding="SAME")) 
    model.add(Flatten())
    model.add(Dense(int(input_dim), activation = 'relu'))
    #model.add(Dense(int(input_dim/8), activation = 'relu'))
    model.add(Dense(out_dim, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    return model

skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y): 
    y_train=to_categorical(y[train])#generate the resonable results
    cv_clf =get_CNN_model(input_dim,out_dim)
    X_train=np.reshape(X[train],(-1,1,input_dim))
    X_test=np.reshape(X[test],(-1,1,input_dim))
    hist=cv_clf.fit(X_train, 
                    y_train,
                    nb_epoch=30)
    y_test=to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    y_score=cv_clf.predict(X_test)#the output of  probability
    yscore=np.vstack((yscore,y_score))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= categorical_probas_to_classes(y_score)
    acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('GTB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
    hist=[]
    cv_clf=[]
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yscore_sum_CNN.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_CNN.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='CNN ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('CNN.csv')