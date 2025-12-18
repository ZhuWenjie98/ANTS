import numpy as np
import sklearn.metrics as sk

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(in_examples, out_examples):
    
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr

def evaluate_all(in_scores, out_scores, label, pred): 
    accuracy = acc(pred, label) # only calculate the in-domain accuracy.
    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))
    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    return [fpr95, auroc, aupr_in, aupr_out, accuracy]
    #return auroc, aupr_in, aupr_out, fpr95

# accuracy
def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc    

## from knn-ood github
# def cal_metric(known, novel, method=None):
#     tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
#     results = dict()
#     # FPR
#     mtype = 'FPR'
#     results[mtype] = fpr_at_tpr95
#
#     # AUROC
#     mtype = 'AUROC'
#     tpr = np.concatenate([[1.], tp/tp[0], [0.]])
#     fpr = np.concatenate([[1.], fp/fp[0], [0.]])
#     results[mtype] = -np.trapz(1.-fpr, tpr)
#
#     # AUIN
#     mtype = 'AUIN'
#     denom = tp+fp
#     denom[denom == 0.] = -1.
#     pin_ind = np.concatenate([[True], denom > 0., [True]])
#     pin = np.concatenate([[.5], tp/denom, [0.]])
#     results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
#
#     # AUOUT
#     mtype = 'AUOUT'
#     denom = tp[0]-tp+fp[0]-fp
#     denom[denom == 0.] = -1.
#     pout_ind = np.concatenate([[True], denom > 0., [True]])
#     pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
#     results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
#
#     return results['AUROC'], results['AUIN'], results['AUOUT'], results['FPR']
#
# def get_curve(known, novel, method=None):
#     tp, fp = dict(), dict()
#     fpr_at_tpr95 = dict()
#
#     known.sort()
#     novel.sort()
#
#     end = np.max([np.max(known), np.max(novel)])
#     start = np.min([np.min(known),np.min(novel)])
#
#     all = np.concatenate((known, novel))
#     all.sort()
#
#     num_k = known.shape[0]
#     num_n = novel.shape[0]
#
#     if method == 'row':
#         threshold = -0.5
#     else:
#         threshold = known[round(0.05 * num_k)]
#
#     tp = -np.ones([num_k+num_n+1], dtype=int)
#     fp = -np.ones([num_k+num_n+1], dtype=int)
#     tp[0], fp[0] = num_k, num_n
#     k, n = 0, 0
#     for l in range(num_k+num_n):
#         if k == num_k:
#             tp[l+1:] = tp[l]
#             fp[l+1:] = np.arange(fp[l]-1, -1, -1)
#             break
#         elif n == num_n:
#             tp[l+1:] = np.arange(tp[l]-1, -1, -1)
#             fp[l+1:] = fp[l]
#             break
#         else:
#             if novel[n] < known[k]:
#                 n += 1
#                 tp[l+1] = tp[l]
#                 fp[l+1] = fp[l] - 1
#             else:
#                 k += 1
#                 tp[l+1] = tp[l] - 1
#                 fp[l+1] = fp[l]
#
#     j = num_k+num_n-1
#     for l in range(num_k+num_n-1):
#         if all[j] == all[j-1]:
#             tp[j] = tp[j+1]
#             fp[j] = fp[j+1]
#         j -= 1
#
#     fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)
#
#     return tp, fp, fpr_at_tpr95
#
# def evaluate_all(in_scores, out_scores):
#     in_examples = in_scores.reshape((-1,))
#     out_examples = out_scores.reshape((-1,))
#     auroc, aupr_in, aupr_out, fpr95 = cal_metric(in_examples, out_examples)
#     return auroc, aupr_in, aupr_out, fpr95