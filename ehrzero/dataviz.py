import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

def plot_confusion_matrix(df, classes = ["0", "1"],
           normalize=False,
           title='Confusion matrix',
           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("F1 score      >> ", sklearn.metrics.f1_score(df["truth"], df["pred_label"]))
    print("ROC AUC score >> ", roc_auc_score(df["truth"], df["pred_label"]))
    cm = sklearn.metrics.confusion_matrix(df["truth"], df["pred_label"])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    
def analyze_class_predictions(df,thr):
    """
        Plot the classwise prediction distributions
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True ,figsize=(10,12))
    sns.distplot(df[df.truth == 0]["predicted_risk"], bins = 30, color = "navy", ax = ax1)
    #ax1.axvline(thr, 'k')
    ax1.set_title("Risks predicted for Negative records")
    sns.distplot(df[df.truth == 1]["predicted_risk"], bins = 30, color = "maroon", ax = ax2)
    ax2.set_title("Risks predicted for Positive records")
    #ax2.axvline(thr, 'k')
    plt.show()
    
def plot_tuning_stats(df, metric = "AUC", min_value_to_show = 0):
    if min_value_to_show:
        df = df[df[metric] > min_value_to_show]
    plt.figure(figsize=(6,6))
    plt.style.use('seaborn-whitegrid')
    FEATURES = list(df.columns)[1:]
    for feat in FEATURES:
        plt.figure(figsize=(5,5))
        plt.plot(df[feat], df[metric], '*')
        plt.title(feat)
        plt.show()