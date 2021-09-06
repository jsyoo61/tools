import sklearn.metrics as metrics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
'confusion_matrix'
]

def plot_confusion_matrix(y_true, y_pred):
    '''Just a wrapper to generate the confusion matrix and the plot'''
    c_matrix = confusion_matrix(y_true, y_pred)
    return return confusion_matrix(c_matrix)

def confusion_matrix(c_matrix):
    '''Plot confusion matrix heatmap'''
    df_cm = pd.DataFrame(c_matrix)
    fig = plt.figure()
    ax = sns.heatmap(df_cm, annot=True)
    ax.set_ylabel('y_true')
    ax.set_xlabel('y_pred')
    return fig, ax

if __name__ == '__main__':
    pass
