import numpy as np
from TDPR import *
# Load the data
val_error_index=np.load("demo/val_error_index.npy")
test_error_index=np.load("demo/test_error_index.npy")
val_pros=np.load("demo/val_pros.npy")
val_labels=np.load("demo/val_labels.npy")
val_infos=np.load("demo/val_infos.npy")
test_pros=np.load("demo/test_pros.npy")
test_labels=np.load("demo/test_labels.npy")
test_infos=np.load("demo/test_infos.npy")
val_features = extract_features(val_pros, val_labels, val_infos)
test_features = extract_features(test_pros, test_labels, test_infos)
val_train_label = np.zeros(len(val_features), dtype=int)
valid_error_index = val_error_index[val_error_index < len(val_train_label)]
print("len(val_train_label):", len(val_train_label))
print("max(val_error_index):", np.max(val_error_index))
assert np.max(val_error_index) < len(val_train_label), "val_error_index contains out-of-bound indices"

val_train_label[valid_error_index] = 1
xgb_rank_params = {
    'objective': 'rank:pairwise',
    'colsample_bytree': 0.5,  # This is the ratio of the number of columns used
    'nthread': -1,
    'eval_metric': 'ndcg',
    'max_depth': 5,
    'min_child_weight': 1,
    # 'subsample': 0.6,
    'learning_rate': 0.05,
    # 'n_estimators':50,
    # 'gamma':0.01,
}
train_data = DMatrix(val_features, label=val_train_label)
rankModel = xgboost.train(xgb_rank_params, train_data)

test_data = DMatrix(test_features)
scores = rankModel.predict(test_data)
test_num = len(test_features)
is_bug = np.zeros(test_num)
is_bug[test_error_index] = 1
index = np.argsort(scores)[::-1]
is_bug = is_bug[index]
print("TDPR:")
print(rauc(is_bug, 100))
print(rauc(is_bug, 200))
print(rauc(is_bug, 500))
print(rauc(is_bug, 1000))
print(rauc(is_bug, test_num))
print(ATRC(is_bug, len(test_error_index)))

deepgini=[]
pros=test_pros[-1,:,:]
for pro in np.array(pros):
    deepgini.append(-np.sum(pro * pro) + 1)
is_bug = np.zeros(test_num)
is_bug[test_error_index] = 1
index = np.argsort(deepgini)[::-1]
is_bug = is_bug[index]
print("DeepGini:")
print(rauc(is_bug, 100))
print(rauc(is_bug, 200))
print(rauc(is_bug, 500))
print(rauc(is_bug, 1000))
print(rauc(is_bug, test_num))
print(ATRC(is_bug, len(test_error_index)))
