import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler



test = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')
X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']
X_test = test.drop(['id'], axis=1)
n_fold = 20
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
repeated_folds = RepeatedStratifiedKFold(n_splits=20, n_repeats=20, random_state=42)

y_train.hist()
plt.show()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.08, solver='liblinear')
oof = np.zeros(len(X_train))
prediction = np.zeros(len(X_test))
scores = []
feature_importance = pd.DataFrame()
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train, y_train)):
        # print('Fold', fold_n, 'started at', time.ctime())
        X_train_folds, X_valid_folds = X_train[train_index], X_train[valid_index]
        y_train_folds, y_valid_folds = y_train[train_index], y_train[valid_index]

        model.fit(X_train_folds, y_train_folds)
        y_pred_valid = model.predict(X_valid_folds)
        score = roc_auc_score(y_valid_folds, y_pred_valid)
        # print(f'Fold {fold_n}. AUC: {score:.4f}.')
        # print('')

        y_pred = model.predict_proba(X_test)[:, 1]
        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(roc_auc_score(y_valid_folds, y_pred_valid))
prediction /= n_fold
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))