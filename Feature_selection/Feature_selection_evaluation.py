import os,sys
import numpy as np
import pandas as pd
import xgboost as xgb
import re
from imblearn.ensemble import EasyEnsemble
from collections import Counter
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from HH import to_categorical,categorical_probas_to_classes,calculate_performace
from sklearn.preprocessing import scale,StandardScaler
from keras.layers import Dense, merge,Input,Dropout
from keras.models import Model
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#xgb_model=xgb.XGBClassifier()
#xgbresult1=xgb_model.fit(shu,label.ravel())
#feature_importance=xgbresult1.feature_importances_
#feature_number=-feature_importance
#H1=np.argsort(feature_number)
#mask=H1[:300]
#train_data=shu[:,mask]

#find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# choose the method
fea_file = sys.argv[1]
# the inprt label

#fea_file='JMI_out.csv'
#labelfile='nr_label.txt'
labelfile=sys.argv[2]

file = open(labelfile,'r')
index=[]
label=[]
for line in file.readlines():
    if(re.match(">",line)):
        array=re.split('>',line)
        index.append(array[1])
    else:
        label.append(int(line[0]))
data={'Index':index,'Label':label}
label = pd.DataFrame(data)
label = label.set_index(['Index'])





train_data=pd.read_csv(fea_file,index_col=0)

X=train_data
y=label

X_resampled_smote, y_resampled_smote = SMOTE(ratio={1:22086}).fit_sample(X,y)
X_resampled_smote, y_resampled_smote = RandomUnderSampler(ratio={0:22086}).fit_sample(X_resampled_smote, y_resampled_smote)
sorted(Counter(y_resampled_smote).items())

data= X_resampled_smote
label=y_resampled_smote
#shu=scale(data_AAindex)
#shu=scale(data)
X=data
y=label
[m,n]=np.shape(X)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
def get_con_model():
    input_1 = Input(shape=(n,), name='Protein')
    protein_input1 = Dense(int(n*2), activation='relu', init='glorot_normal', name='High_dim_feature_1')(input_1)
    protein_input1=Dropout(0.5)(protein_input1)
    protein_input1 = Dense(int(n), activation='relu', init='glorot_normal', name='High_dim_feature_2')(protein_input1)
    protein_input1=Dropout(0.5)(protein_input1)
    protein_input1 = Dense(int(n/2), activation='relu', init='glorot_normal', name='High_dim_feature_3')(protein_input1)
    protein_input1=Dropout(0.5)(protein_input1)
    output = Dense(int(n/4), activation='relu', init='glorot_normal', name='High_dim_feature')(protein_input1)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(input=input_1, output=outputs)
    #sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model
skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y): 
    y_train=to_categorical(y[train])#generate the resonable results
    cv_clf =get_con_model()
    hist=cv_clf.fit(X[train], 
                    y_train,
                    nb_epoch=50)
    y_test=to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    y_score=cv_clf.predict(X[test])#the output of  probability
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
yscore_sum.to_csv('yscore'+fea_file+'.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest'+fea_file+'.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='DNN ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('result'+fea_file+'.csv')