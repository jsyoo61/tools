import sklearn.metrics as metrics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
'confusion_matrix',
'plot_confusion_matrix',
]

def plot_confusion_matrix(y_true, y_pred, ax=None, **kwargs):
    '''Just a wrapper to generate the confusion matrix and the plot'''
    c_matrix = metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix(c_matrix, ax=ax)

def confusion_matrix(c_matrix, ax=None, **kwargs):
    '''Plot confusion matrix heatmap'''
    df_cm = pd.DataFrame(c_matrix)
    if ax is None:
        ax = sns.heatmap(df_cm, annot=True, **kwargs)
    else:
        ax = sns.heatmap(df_cm, annot=True, ax=ax, **kwargs)
    ax.set_ylabel('y_true')
    ax.set_xlabel('y_pred')
    return ax

if __name__ == '__main__':
    pass
