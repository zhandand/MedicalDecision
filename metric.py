import numpy as np
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score

# 对所有病人求jaccard
def jaccard(y_gt, y_pred):
	score = []
	for b in range(y_gt.shape[0]):
		target = np.where(y_gt[b] == 1)[0]
		out_list = np.where(y_pred[b] == 1)[0]
		inter = set(out_list) & set(target)
		union = set(out_list) | set(target)
		jaccard_score = 0 if union == 0 else len(inter) / len(union)
		score.append(jaccard_score)
	return np.mean(score)

# 精确率
def average_prc(y_gt, y_pred):
    # 预测对的占预测药物总数的大小
	score = []
	for b in range(y_gt.shape[0]):
		target = np.where(y_gt[b] == 1)[0]
		out_list = np.where(y_pred[b] == 1)[0]
		inter = set(out_list) & set(target)
		prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
		score.append(prc_score)
	return score

# 召回率
def average_recall(y_gt, y_pred):
    # 预测对的占ground truth总数的大小
	score = []
	for b in range(y_gt.shape[0]):
		target = np.where(y_gt[b] == 1)[0]
		out_list = np.where(y_pred[b] == 1)[0]
		inter = set(out_list) & set(target)
		recall_score = 0 if len(target) == 0 else len(inter) / len(target)
		score.append(recall_score)
	return score

# f1
def average_f1(y_gt, y_pred):
    prc, recall = [],[]
    prc = average_prc(y_gt, y_pred)
    recall = average_recall(y_gt, y_pred)
    score = []
    for idx in range(len(prc)):
        if prc[idx] + recall[idx] == 0:
            score.append(0)
        else:
            score.append( 
                2*prc[idx]*recall[idx] / (prc[idx] + recall[idx]))
    return np.mean(score)

def f1(y_gt, y_pred):
	all_micro = []
	for b in range(y_gt.shape[0]):
		all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
	return np.mean(all_micro)

def roc_auc(y_gt, y_prob):
	all_micro = []
	for b in range(len(y_gt)):
		all_micro.append(roc_auc_score(
			y_gt[b], y_prob[b], average='macro'))
	return np.mean(all_micro)

# PRAUC
def precision_auc(y_gt, y_prob):
	all_micro = []
	for b in range(len(y_gt)):
		all_micro.append(average_precision_score(
			y_gt[b], y_prob[b], average='macro'))
	return np.mean(all_micro)

def precision_at_k(y_gt, y_prob, k=3):
	precision = 0
	sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
	for i in range(len(y_gt)):
		TP = 0
		for j in range(len(sort_index[i])):
			if y_gt[i, sort_index[i, j]] == 1:
				TP += 1
		precision += TP / len(sort_index[i])
	return precision / len(y_gt)

# auc = roc_auc(y_gt, y_prob)
# p_1 = precision_at_k(y_gt, y_prob, k=1)
# p_3 = precision_at_k(y_gt, y_prob, k=3)
# p_5 = precision_at_k(y_gt, y_prob, k=5)
# f1 = f1(y_gt, y_pred)
# prauc = precision_auc(y_gt, y_prob)
# ja = jaccard(y_gt, y_pred)
# avg_prc = average_prc(y_gt, y_pred)
# avg_recall = average_recall(y_gt, y_pred)
# avg_f1 = average_f1(avg_prc, avg_recall)
# ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)
