import sys,os
import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import math
import scipy.spatial as ss
from scipy.sparse import csc_matrix
from scipy.sparse import diags
from scipy.special import digamma
from math import log
import numpy.random as nr
from skrebate import ReliefF
import numpy.matlib
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA

#find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

#choose the method
#option = sys.argv[1]
#option = "1"
# read number of feature to select
#selected_number = int(sys.argv[2])
selected_number=300

# read feature vectors
#featurefile1 = sys.argv[3]
#featurefile2 = sys.argv[4]

featurefile1 = 'NR_zheng.csv'
featurefile2 = 'NR_fu.csv'
#featurefile='RNA_Psednc_PSTNP_TNC_DNC_MonoKGap.csv'
feature1 = pd.read_csv(featurefile1,header=None, float_precision='round_trip')
feature2 = pd.read_csv(featurefile2,header=None, float_precision='round_trip')

feature=pd.concat([feature1,feature2])
#feature = pd.read_csv("DNA_Kmer_Psednc_DAC.csv",header=None,index_col=0,float_precision='round_trip')
feature_1=scale(feature)
feature=pd.DataFrame(feature_1)

# read feature labels
#labelfile = sys.argv[4]
labelfile='nr_label.txt'

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

#- Feature selection methods ---------------------------------------------------------------------

# Feature selection Lasso    
def Lasso_selection(X,y,alpha=0.01):
    lasso_model=Lasso(alpha=alpha)
    lasso_ = lasso_model.fit(X,y.values.ravel())
    lassoresult = np.nonzero(lasso_.coef_)
    lassoresult = X[X.columns[lassoresult]]
    lassoresult.to_csv("Lasso_out.csv")
    return None

# Feature selection Elastic-Net
def Elastic_selection(X,y,alpha =0.1,l1_ratio=0.1):
    enet=ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic = enet.fit(X,y.values.ravel())
    elasticresult=np.nonzero(elastic.coef_)
    elasticresult = X[X.columns[elasticresult]]
    elasticresult.to_csv("ElasticNet_out.csv")
    return None

# Feature selection MRMR
def entropy(x, k=3, base=2):    
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    d = len(x[0])
    N = len(x)
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
    tree = ss.cKDTree(x)
    nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in x]
    const = digamma(N)-digamma(k) + d*log(2)
    return (const + d*np.mean(map(log, nn)))/log(base)

def entropyd(sx, base=2):  # Discrete estimators
    return entropyfromprobs(hist(sx), base=base)

def midd(x, y):
    return -entropyd(list(zip(x, y)))+entropyd(x)+entropyd(y)

def cmidd(x, y, z):
    return entropyd(list(zip(y, z)))+entropyd(list(zip(x, z)))-entropyd(list(zip(x, y, z)))-entropyd(z)

def hist(sx):
    # Histogram from list of samples
    d = dict()
    #d=list()
    for s in sx:
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d.values())

def entropyfromprobs(probs, base=2):
    # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
    return -sum(map(elog, probs))/log(base)

def elog(x):
    # for entropy, 0 log 0 = 0. but we get an error for putting log 0
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)

def micd(x, y, k=3, base=2, warning=True): # Mixed estimators
    overallentropy = entropy(x, k, base)
    n = len(y)
    word_dict = dict()
    for sample in y:
        word_dict[sample] = word_dict.get(sample, 0) + 1./n
    yvals = list(set(word_dict.keys()))
    mi = overallentropy
    for yval in yvals:
        xgiveny = [x[i] for i in range(n) if y[i] == yval]
        if k <= len(xgiveny) - 1:
            mi -= word_dict[yval]*entropy(xgiveny, k, base)
        else:
            if warning:
                print("Warning, after conditioning, on y={0} insufficient data. Assuming maximal entropy in this case.".format(yval))
            mi -= word_dict[yval]*overallentropy
    return mi  # units already applied
	
def zip2(*args):
    return [sum(sublist, []) for sublist in zip(*args)]

def lcsi(X, y, **kwargs):    
    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMI = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False
    # initialize the parameters
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1
    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]
        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break
        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        if 'function_name' in kwargs.keys():
            if kwargs['function_name'] == 'MRMR':
                beta = 1.0 / len(F)
            elif kwargs['function_name'] == 'JMI':
                beta = 1.0 / len(F)
                gamma = 1.0 / len(F)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += midd(f_select, f)
                t3[i] += cmidd(f_select, f, y)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - beta*t2[i] + gamma*t3[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]
    return np.array(F), np.array(J_CMI), np.array(MIfy)


# Feature selection MRMD
def calcE(X,coli,colj):
    sum1 = np.sum((X[:,coli]-X[:,colj])**2)  
    return math.sqrt(sum1)

def Euclidean(X,n):
    Euclideandata=np.zeros([n,n])    
    for i in range(n):
        for j in range(n):
            Euclideandata[i,j]=calcE(X,i,j)
            Euclideandata[j,i]=Euclideandata[i,j]
    Euclidean_distance=[]

    for i in range(n):
        sum1 = np.sum(Euclideandata[i,:])
        Euclidean_distance.append(sum1/n)
    return Euclidean_distance

def varience(data,avg1,col1,avg2,col2):
    return np.average((data[:,col1]-avg1)*(data[:,col2]-avg2))

def Person(X,y,n):
    feaNum=n
    #label_num=len(y[0,:])
    label_num=1
    PersonData=np.zeros([n])
    for i in range(feaNum):
        for j in range(feaNum,feaNum+label_num):
            #print('. ', end='')
            average1 = np.average(X[:,i])
            average2 = np.average(y)
            yn=(X.shape)[0]
            y=y.reshape((yn,1))
            dataset = np.concatenate((X,y),axis=1)
            numerator = varience(dataset, average1, i, average2, j);
            denominator = math.sqrt(
                varience(dataset, average1, i, average1, i) * varience(dataset, average2, j, average2, j));
            if (abs(denominator) < (1E-10)):
                PersonData[i]=0
            else:
                PersonData[i]=abs(numerator/denominator)
    return list(PersonData)

def mrmd(X,y,n_selected_features=10):
    n=X.shape[1]
    e=Euclidean(X,n)
    p = Person(X,y,n)
    mrmrValue=[]
    for i,j in zip(p,e):
        mrmrValue.append(i+j)
    mrmr_max=max(mrmrValue)
    features_name=np.array(range(n))
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name,mrmrValue)]   
    mrmd_order=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  
    mrmd_order =[int(x[0]) for x in mrmd_order]
    mrmd_end=mrmd_order[:n_selected_features]
    return mrmd_end

# Feature selection ReliefF
def ReliefF_Method(X,y,n):
    X=np.array(X)
    y=np.array(y)
    y = y[:, 0]
    clf = ReliefF(n_features_to_select=n, n_neighbors=50)
    Reresult = clf.fit_transform(X,y)
    Reresult = pd.DataFrame(Reresult)
    Reresult.to_csv("ReliefF_out.csv")
    return None

# Feature selection gini index
def gini_index(X, y):
    n_samples, n_features = X.shape
    # initialize gini_index for all features to be 0.5
    gini = np.ones(n_features) * 0.5
    # For i-th feature we define fi = x[:,i] ,v include all unique values in fi
    for i in range(n_features):
        v = np.unique(X[:, i])
        for j in range(len(v)):
            # left_y contains labels of instances whose i-th feature value is less than or equal to v[j]
            left_y = y[X[:, i] <= v[j]]
            # right_y contains labels of instances whose i-th feature value is larger than v[j]
            right_y = y[X[:, i] > v[j]]
            # gini_left is sum of square of probability of occurrence of v[i] in left_y
            # gini_right is sum of square of probability of occurrence of v[i] in right_y
            gini_left = 0
            gini_right = 0
            for k in range(np.min(y), np.max(y)+1):
                if len(left_y) != 0:
                    # t1_left is probability of occurrence of k in left_y
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    t2_left = np.power(t1_left, 2)
                    gini_left += t2_left
                if len(right_y) != 0:
                    # t1_right is probability of occurrence of k in left_y
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    t2_right = np.power(t1_right, 2)
                    gini_right += t2_right
            gini_left = 1 - gini_left
            gini_right = 1 - gini_right
            # weighted average of len(left_y) and len(right_y)
            t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)
            # compute the gini_index for the i-th feature
            value = np.true_divide(t1_gini, len(y))
            if value < gini[i]:
                gini[i] = value
    return gini

def feature_ranking(W):
    idx = np.argsort(W)
    return idx


def feature_ranking2(score, **kwargs):
    if 'style' not in kwargs:
        kwargs['style'] = 0
    style = kwargs['style']

    # if style = -1 or 0, ranking features in descending order, the higher the score, the more important the feature is
    if style == -1 or style == 0:
        idx = np.argsort(score, 0)
        return idx[::-1]
    # if style != -1 and 0, ranking features in ascending order, the lower the score, the more important the feature is
    elif style != -1 and style != 0:
        idx = np.argsort(score, 0)
        return idx


# Feature selection IG
def calProb(array):
	myProb = {}
	myClass = set(array)
	for i in myClass:
		myProb[i] = array.count(i) / len(array)
	return myProb

def jointProb(newArray, labels):
	myJointProb = {}
	for i in range(len(labels)):
		myJointProb[str(newArray[i]) + '-' + str(labels[i])] = myJointProb.get(str(newArray[i]) + '-' + str(labels[i]), 0) + 1

	for key in myJointProb:
		myJointProb[key] = myJointProb[key] / len(labels)
	return myJointProb
def IG(encodings,labelfile,k):
    encoding=np.array(encodings)
    sample=encoding[:,0]
    data=encoding[:,:]
    shape=data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)
    samples=[i for i in sample]
    file = open(labelfile,'r')
    file.readline()
    for line in file.readlines():
        records=line[:]
    myDict = {}
    try:
	    for i in records:
		     array = i.rstrip().split() if i.strip() != '' else None
		     myDict[array[0]] = int(array[1])
    except IndexError as e:
        print(e)
    labels = []
    for i in samples:
	     labels.append(myDict.get(i, 0))

    dataShape = data.shape
    features=range(dataShape[1])

    if dataShape[0] != len(labels):
	    print('Error: inconsistent data shape with sample number.')
    probY = calProb(labels)
    myFea = {}
    binBox = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(len(features)):
	     array = data[:, i]
	     newArray = list(pd.cut(array, len(binBox), labels= binBox))
	     probX = calProb(newArray)
	     probXY = jointProb(newArray, labels)
	     HX = -1 * sum([p * math.log(p, 2) for p in probX.values()])
	     HXY = 0
	     for y in probY.keys():
		     for x in probX.keys():
			     if str(x) + '-' + str(y) in probXY:
				     HXY = HXY + (probXY[str(x) + '-' + str(y)] * math.log(probXY[str(x) + '-' + str(y)] / probY[y], 2))
	     myFea[features[i]] = HX + HXY
    res=[]
    for key in sorted(myFea.items(), key=lambda item:item[1], reverse=True):
	    res.append([key[0], '{0:.3f}'.format(myFea[key[0]])]) 
    res=np.array(res)
    importance=res[:,0]
    feature_=np.array([float(i) for i in importance])
    H1=np.argsort(-feature_)
    mask=H1[:k].astype(int)
    return mask

#- main function ----------------------------------------------------------------------

Lasso_selection(feature,label,alpha=0.001)
# Feature selection Elastic-Net
Elastic_selection(feature,label,alpha =0.1,l1_ratio=0.05)

#Feature selection MRMD
X=np.array(feature)
y=np.array(label)
y = y[:, 0]
MRMDresult=mrmd(X,y,n_selected_features=selected_number)
MRMDresult = feature[feature.columns[MRMDresult]]
MRMDresult.to_csv("MRMD_out.csv")
# Feature selection ReliefF
ReliefF_Method(feature,label,selected_number)
#
X=np.array(feature)
y=np.asarray(label)
y = y[:, 0]
score = gini_index(X, y)
# rank features in descending order according to score
idx =feature_ranking(score)
giniresult = feature[feature.columns[idx[0:selected_number]]]
giniresult.to_csv("GINI_INDEX_out.csv")
#

X=np.array(feature)
y=np.asarray(label)
y = y[:, 0] 
# specify the second ranking function which uses all except the 1st eigenvalue
kwargs = {'style': 0}
X=np.array(feature)
y=np.asarray(label)
y = y[:, 0]
IGresult=IG(feature,labelfile,selected_number)
IGresult = feature[feature.columns[IGresult]]
IGresult.to_csv("IG_out.csv")

    
