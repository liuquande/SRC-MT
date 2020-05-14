# encoding: utf-8
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#, sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
from sklearn.metrics.ranking import roc_auc_score


N_CLASSES = 7
CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']

def compute_AUCs(gt, pred, competition=True):
    """
    Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    indexes = range(len(CLASS_NAMES))
    
    for i in indexes:
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            AUROCs.append(0)
    return AUROCs


def compute_metrics(gt, pred, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    AUROCs, Accus, Senss, Recas, Specs = [], [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    # if cfg.uncertainty == 'U-Zeros':
    #     gt_np[np.where(gt_np==-1)] = 0
    # if cfg.uncertainty == 'U-Ones':
    #     gt_np[np.where(gt_np==-1)] = 1
    pred_np = pred.cpu().detach().numpy()
    THRESH = 0.18
    #     indexes = TARGET_INDEXES if competition else range(N_CLASSES)
    #indexes = range(n_classes)
    
#     pdb.set_trace()
    indexes = range(len(CLASS_NAMES))
    
    for i, cls in enumerate(indexes):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            AUROCs.append(0)
        
        try:
            Accus.append(accuracy_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)
        
        try:
            Senss.append(sensitivity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing precision for {}.'.format(i))
            Senss.append(0)
        

        try:
            Specs.append(specificity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Specs.append(0)
    
    return AUROCs, Accus, Senss, Specs

def compute_metrics_test(gt, pred, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False, 
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    AUROCs, Accus, Senss, Specs, Pre, F1 = [], [], [], [], [], []
    gt_np = gt.cpu().detach().numpy()
    # if cfg.uncertainty == 'U-Zeros':
    #     gt_np[np.where(gt_np==-1)] = 0
    # if cfg.uncertainty == 'U-Ones':
    #     gt_np[np.where(gt_np==-1)] = 1
    pred_np = pred.cpu().detach().numpy()
    THRESH = 0.18
    #     indexes = TARGET_INDEXES if competition else range(N_CLASSES)
    #indexes = range(n_classes)
    
#     pdb.set_trace()
    indexes = range(len(CLASS_NAMES))
    
    for i, cls in enumerate(indexes):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            AUROCs.append(0)
        
        try:
            Accus.append(accuracy_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            Accus.append(0)
        
        try:
            Senss.append(sensitivity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing precision for {}.'.format(i))
            Senss.append(0)
        

        try:
            Specs.append(specificity_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Specs.append(0)

        try:
            Pre.append(precision_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            Pre.append(0)
    
        try:
            F1.append(f1_score(gt_np[:, i], (pred_np[:, i]>=THRESH)))
        except ValueError:
            print('Error in computing F1-score for {}.'.format(i))
            F1.append(0)
    
    return AUROCs, Accus, Senss, Specs, Pre, F1
