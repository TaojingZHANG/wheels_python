from sklearn.datasets import make_classification
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from testLoadGBM import InferenceLightGBM

X,y = make_classification(n_classes=2,n_samples=200,random_state=100,n_features=10)
sex_list = ['Male','Female']
age_list = ['Youth','Adult','Elder']
X = pd.DataFrame(X,columns=['Col_{}'.format(i) for i in range(10)])
for i in range(200):
    X.loc[i,'Sex'] = np.random.choice(sex_list)
for i in range(200):
    X.loc[i,'Age'] = np.random.choice(age_list)
X['Sex'] = X['Sex'].astype('category')
X['Age'] = X['Age'].astype('category')
dtrain = lgb.Dataset(X,y,feature_name='auto',categorical_feature='auto',free_raw_data=False)

booster_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
evals_result = {}
gbm = lgb.train(booster_params,
                num_boost_round=200,
                train_set=dtrain,
                valid_sets=[dtrain],
                valid_names=['tr'],
                evals_result=evals_result,
                verbose_eval=50,
                early_stopping_rounds=10,
                )
model_json = gbm.dump_model()
model_json['feature_names'] = list(dtrain.data.columns)

with open("sample_model.json", 'w') as json_file:
    json.dump(model_json, json_file, ensure_ascii=False)

cat_features = [column for column in dtrain.data.columns if hasattr(dtrain.data[column], 'cat')]
category_dict = dict()
for cat_feature in cat_features:
    category_dict[cat_feature] = {v: k for k, v in enumerate(list(dtrain.data[cat_feature].cat.categories))}

with open("category_feature_map.json", 'w') as json_file:
    json.dump(category_dict, json_file, ensure_ascii=False)



#######test

inf_lgb = InferenceLightGBM("sample_model.json", "category_feature_map.json")
sample = dtrain.data
result_json = inf_lgb.predict(sample)
result_gbm = gbm.predict(sample)

diffrence = result_json.values - result_gbm
print(diffrence)