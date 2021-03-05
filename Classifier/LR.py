import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from HH import to_categorical,categorical_probas_to_classes,calculate_performace
from sklearn.preprocessing import scale,StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

shu1 = pd.read_csv("NR_zheng.csv",header=None)
shu2 = pd.read_csv("NR_fu.csv",header=None)
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

X_resampled_smote, y_resampled_smote = SMOTE(ratio={1:540}).fit_sample(X,y)
X_resampled_smote, y_resampled_smote = RandomUnderSampler(ratio={0:540}).fit_sample(X_resampled_smote, y_resampled_smote)
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
cv_clf = LogisticRegression()
skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y): 
    X_train_enc=cv_clf.fit(X[train], y[train])
    y_score=cv_clf.predict_proba(X[test])
    yscore=np.vstack((yscore,y_score))
    y_test=to_categorical(y[test]) 
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('LR:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
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
yscore_sum.to_csv('yscore_sum_LR.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_LR.csv')
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
data_csv.to_csv('LR.csv')