##DNN-DTIs

DNN-DTIs: improved drug-target interactions prediction using XGBoost feature selection and deep neural network

###DNN-DTIs uses the following dependencies:
* python 3.6 
* numpy
* pandas
* imblearn
* scikit-learn
* tensorflow
* keras

###Guiding principles:

**The Dataset file contains the gold standard dataset, Kuang dataset and network dataset.

**Feature extraction
   CTDC.py, CTDD.py and CTDT.py are the implementation of composition, transition and distribution.
   CTriad.py is the implementation of conjoint triad.
   PAAC.py is the implementation of pseudo amino acid composition.
   NMBroto.py is the implementation of Moreau-Broto autocorrelation.
   PsePSSM.m is the implementation of pseudo position-specific scoring matrix.
   structure_B.py, structure_C.py and structure_D.py are the implementation of structure feature.
  
** Feature selection:
   XGBoost.py represents XGBoost feature selection.
   feature_selection.py includes IG, GINI, MRMD, LASSO and EN.
   Feature_selection_evaluation.py can output the evaluation indicators based on DNN.

** Classifier:
   DNN.py is the implementation of deep neural network.
   CNN.py is the implementation of convolutional neural network.
   LSTM.py is the implementation of long and short term memory neural network.
   AdaBoost.py is the implementation of AdaBoost.
   KNN.py is the implementation of K nearest neighbor.
   LR.py is the implementation of logistic regression.
   RF.py is the implementation of random forest.
   SVM.py is the implementation of support vector machine.