from pyltr_model import *

training_set = "90_train"
validation_set = "90_valid"
test_set = "test_set_VU_DM"
model_name = f"model_{training_set}3"
output_name = f"output_{model_name}_{test_set}"

params = dict(n_estimators=1500,
              learning_rate=0.002,
              max_features=0.5,
              subsample=1.0,
              min_samples_split=16,
              max_depth=4,
              query_subsample=1.0,
              max_leaf_nodes=None,
              min_samples_leaf=64)

train_and_save(training_set, validation_set, model_name, params, 3)
predict_generate(test_set, model_name, output_name)