import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    # __import__("ipdb").set_trace()
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def split_cluster_acc_v2(y_true, y_pred, mask, args=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        args: arguments, not used

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return {
        'All': total_acc, 
        'Old': old_acc, 
        'New': new_acc,
    }



def split_cluster_acc_v2_balanced(y_true, y_pred, mask, args=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        args: arguments, not used

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc[total_acc == np.inf] = 0
    old_acc[old_acc == np.inf] = 0
    new_acc[new_acc == np.inf] = 0
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()
    return {
        'All': total_acc, 
        'Old': old_acc, 
        'New': new_acc,
    }

def split_cluster_acc_v2_level(y_true, y_pred, mask, args=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        args: arguments 

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
    
    hard_classes_gt = set(args.target_transform(item) for item in args.hard_classes)
    medium_classes_gt = set(args.target_transform(item) for item in args.medium_classes)
    easy_classes_gt = set(args.target_transform(item) for item in args.easy_classes)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    
    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc[total_acc == np.inf] = 0
    old_acc[old_acc == np.inf] = 0
    new_acc[new_acc == np.inf] = 0
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()

    # TODO: abstract this acc calculation
    hard_acc = np.zeros(len(hard_classes_gt))
    total_hard_instances = np.zeros(len(hard_classes_gt))
    for idx, i in enumerate(hard_classes_gt):
        hard_acc[idx] += w[ind_map[i], i]
        total_hard_instances[idx] += sum(w[:, i])

    medium_acc = np.zeros(len(medium_classes_gt))
    total_medium_instances = np.zeros(len(medium_classes_gt))
    for idx, i in enumerate(medium_classes_gt):
        medium_acc[idx] += w[ind_map[i], i]
        total_medium_instances[idx] += sum(w[:, i])
        
    easy_acc = np.zeros(len(easy_classes_gt))
    total_easy_instances = np.zeros(len(easy_classes_gt))
    for idx, i in enumerate(easy_classes_gt):
        easy_acc[idx] += w[ind_map[i], i]
        total_easy_instances[idx] += sum(w[:, i])

    hard_acc /= total_hard_instances
    medium_acc /= total_medium_instances
    easy_acc /= total_easy_instances
    hard_acc[hard_acc == np.inf] = 0
    medium_acc[medium_acc == np.inf] = 0
    easy_acc[easy_acc == np.inf] = 0
    hard_acc, medium_acc, easy_acc = hard_acc.mean(), medium_acc.mean(), easy_acc.mean()

    return {
        'All': total_acc, 
        'Old': old_acc, 
        'New': new_acc,
        'Hard': hard_acc,
        'Medium': medium_acc,
        'Easy': easy_acc,
    }

def split_cluster_acc_v2_overlap(y_true, y_pred, mask, args=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        args: arguments 

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
    
    overlap_classes_gt = set(args.target_transform(item) for item in args.overlap_cls)
    nonoverlap_classes_gt = set(args.target_transform(item) for item in set(args.train_classes) - set(args.overlap_cls))

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    
    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc[total_acc == np.inf] = 0
    old_acc[old_acc == np.inf] = 0
    new_acc[new_acc == np.inf] = 0
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()

    overlap_acc = np.zeros(len(overlap_classes_gt))
    total_overlap_instances = np.zeros(len(overlap_classes_gt))
    for idx, i in enumerate(overlap_classes_gt):
        overlap_acc[idx] += w[ind_map[i], i]
        total_overlap_instances[idx] += sum(w[:, i])

    nonoverlap_acc = np.zeros(len(nonoverlap_classes_gt))
    total_nonoverlap_instances = np.zeros(len(nonoverlap_classes_gt))
    for idx, i in enumerate(nonoverlap_classes_gt):
        nonoverlap_acc[idx] += w[ind_map[i], i]
        total_nonoverlap_instances[idx] += sum(w[:, i])
        
    overlap_acc /= total_overlap_instances
    try:
        nonoverlap_acc /= total_nonoverlap_instances
    except:
        nonoverlap_acc = np.array([0])
        args.logger.info('Using nonoverlap on training unlabelled set')
    overlap_acc[overlap_acc == np.inf] = 0
    nonoverlap_acc[nonoverlap_acc == np.inf] = 0
    overlap_acc, nonoverlap_acc = overlap_acc.mean(), nonoverlap_acc.mean()

    return {
        'All': total_acc, 
        'Old': old_acc, 
        'New': new_acc,
        'Overlap': overlap_acc,
        'Nonoverlap': nonoverlap_acc,
    }

def split_cluster_acc_v2_incremental(y_true, y_pred, mask, args=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: will be ignored
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        args: arguments 
            will use class_in_labelled and class_in_unlabelled

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    
    # args.classes_in_labelled
    # args.classes_in_unlabelled
    
    # overlap_classes_gt = set(args.target_transform(item) for item in args.overlap_cls)
    # nonoverlap_classes_gt = set(args.target_transform(item) for item in set(args.train_classes) - set(args.overlap_cls))
    old_classes_gt = set(args.target_transform(item) for item in args.classes_in_labelled)
    new_classes_gt = set(args.target_transform(item) for item in args.classes_in_unlabelled)

    assert y_pred.size == y_true.size
    D = args.mlp_out_dim
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    
    old_acc = np.zeros(D)
    total_old_instances = np.zeros(D)
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(D)
    total_new_instances = np.zeros(D)
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    # total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_correct = np.array([correct / num_img for correct, num_img in zip(old_acc, total_old_instances) if num_img > 0])
    new_correct = np.array([correct / num_img for correct, num_img in zip(new_acc, total_new_instances) if num_img > 0])
    
    total_correct = np.array([correct / num_img for correct, num_img in zip(np.concatenate([old_acc, new_acc]), 
                                                         np.concatenate([total_old_instances + total_new_instances])) if num_img > 0])

    old_acc, new_acc = old_correct.mean(), new_correct.mean()
    total_acc = total_correct.mean()

    return {
        'All': total_acc, 
        'Old': old_acc, 
        'New': new_acc,
    }

def split_cluster_acc_v2_uni(y_true, y_pred, mask, args=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        args: arguments 

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
    
    hard_classes_gt = set(args.target_transform(item) for item in args.hard_classes)
    medium_classes_gt = set(args.target_transform(item) for item in args.medium_classes)
    easy_classes_gt = set(args.target_transform(item) for item in args.easy_classes)
    
    overlap_classes_gt = set(args.target_transform(item) for item in args.overlap_cls)
    nonoverlap_classes_gt = set(args.target_transform(item) for item in set(args.train_classes) - set(args.overlap_cls))

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    
    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()

    # TODO: abstract this acc calculation
    hard_acc = np.zeros(len(hard_classes_gt))
    total_hard_instances = np.zeros(len(hard_classes_gt))
    for idx, i in enumerate(hard_classes_gt):
        hard_acc[idx] += w[ind_map[i], i]
        total_hard_instances[idx] += sum(w[:, i])

    medium_acc = np.zeros(len(medium_classes_gt))
    total_medium_instances = np.zeros(len(medium_classes_gt))
    for idx, i in enumerate(medium_classes_gt):
        medium_acc[idx] += w[ind_map[i], i]
        total_medium_instances[idx] += sum(w[:, i])
        
    easy_acc = np.zeros(len(easy_classes_gt))
    total_easy_instances = np.zeros(len(easy_classes_gt))
    for idx, i in enumerate(easy_classes_gt):
        easy_acc[idx] += w[ind_map[i], i]
        total_easy_instances[idx] += sum(w[:, i])

    hard_acc /= total_hard_instances
    medium_acc /= total_medium_instances
    easy_acc /= total_easy_instances
    hard_acc, medium_acc, easy_acc = hard_acc.mean(), medium_acc.mean(), easy_acc.mean()


    overlap_acc = np.zeros(len(overlap_classes_gt))
    total_overlap_instances = np.zeros(len(overlap_classes_gt))
    for idx, i in enumerate(overlap_classes_gt):
        overlap_acc[idx] += w[ind_map[i], i]
        total_overlap_instances[idx] += sum(w[:, i])

    nonoverlap_acc = np.zeros(len(nonoverlap_classes_gt))
    total_nonoverlap_instances = np.zeros(len(nonoverlap_classes_gt))
    for idx, i in enumerate(nonoverlap_classes_gt):
        nonoverlap_acc[idx] += w[ind_map[i], i]
        total_nonoverlap_instances[idx] += sum(w[:, i])
        
    overlap_acc /= total_overlap_instances
    try:
        nonoverlap_acc /= total_nonoverlap_instances
    except:
        nonoverlap_acc = np.array([0])
        args.logger.info('Using nonoverlap on training unlabelled set')
    overlap_acc, nonoverlap_acc = overlap_acc.mean(), nonoverlap_acc.mean()

    return {
        'All': total_acc, 
        'Old': old_acc, 
        'New': new_acc,
        'Hard': hard_acc,
        'Medium': medium_acc,
        'Easy': easy_acc,
        'Overlap': overlap_acc,
        'Nonoverlap': nonoverlap_acc,
    }

EVAL_FUNCS = {
    'v2': split_cluster_acc_v2,
    'v2b': split_cluster_acc_v2_balanced,
    'v2l': split_cluster_acc_v2_level,
    'v2o': split_cluster_acc_v2_overlap,
    'v2u': split_cluster_acc_v2_uni,
    'v2i': split_cluster_acc_v2_incremental,
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs, save_name, T=None,
                        print_output=True, args=None):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        output_dict = acc_f(y_true, y_pred, mask, args)
        log_name = f'{save_name}_{f_name}'

        if i == 0:
            # to_return = (output_dict['All'], output_dict['Old'], output_dict['New'])
            to_return = output_dict

        if print_output:
            print_str = f'Epoch {T}, {log_name}: ' # All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            for item in output_dict:
                print_str += f'{item} {output_dict[item]:.4f} | '

            try:
                args.logger.info(print_str)
            except:
                print(print_str)

    return to_return