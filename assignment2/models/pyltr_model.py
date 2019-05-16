import csv
import os
import pickle

import pandas as pd
import pyltr

from data_preprocess import prepare_for_mart


def train_and_save():
    val_dataset = "tiny_train"
    dataset = "training_set_VU_DM"
    dataset = "tenth_train"

    if not os.path.exists(f"{dataset}.pkl"):
        features, relevancies, search_ids, prop_ids = prepare_for_mart(
            path=f"../data/{dataset}.csv", is_train_set=True)

        with open(f"{dataset}.pkl", "wb") as f:
            pickle.dump((features, relevancies, search_ids, prop_ids), f)
    else:
        with open(f"{dataset}.pkl", "rb") as f:
            features, relevancies, search_ids, _ = pickle.load(f)


    if not os.path.exists(f"{val_dataset}.pkl"):
        val_features, val_relevancies, val_search_ids, val_prop_ids = prepare_for_mart(
            path=f"../data/{val_dataset}.csv", is_train_set=True)

        with open(f"{val_dataset}.pkl", "wb") as f:
            pickle.dump((val_features, val_relevancies, val_search_ids, val_prop_ids), f)
    else:
        with open(f"{val_dataset}.pkl", "rb") as f:
            val_features, val_relevancies, val_search_ids, _ = pickle.load(f)

    metric = pyltr.metrics.NDCG(k=10)

    # Only needed if you want to perform validation (early stopping & trimming)
    monitor = pyltr.models.monitors.ValidationMonitor(
        val_features, val_relevancies, val_search_ids, metric=metric, stop_after=250)

    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=1000,
        learning_rate=0.02,
        max_features=0.7,
        subsample=1.0,
        min_samples_split=2,
        max_depth=4,
        query_subsample=0.5,
        max_leaf_nodes=25,
        min_samples_leaf=64,
        verbose=1,
    )
    with open("training_curves.csv", 'w') as f:
        blah = csv.writer(f, delimiter=';')
        blah.writerow(['Iter', 'Train score', 'OOB Improve', 'remaining_time', 'monitor_output'])

    model.fit(features, relevancies, search_ids, monitor=monitor)

    with open(f"model_{dataset}.pkl", "wb") as f:
        pickle.dump(model, f)


def predict_generate():
    dataset = "test_set_VU_DM"
    # dataset = "tiny_test"

    if not os.path.exists(f"{dataset}.pkl"):
        features, relevancies, search_ids, prop_ids = prepare_for_mart(
            path=f"../data/{dataset}.csv", is_train_set=False)

        with open(f"{dataset}.pkl", "wb") as f:
            pickle.dump((features, relevancies, search_ids, prop_ids), f)
    else:
        with open(f"{dataset}.pkl", "rb") as f:
            features, relevancies, search_ids, prop_ids = pickle.load(f)

    with open(f"model_tenth_train.pkl", "rb") as f:
        model: pyltr.models.LambdaMART = pickle.load(f)

    Ey = model.predict(features)

    df = pd.DataFrame({'relevancies': Ey,
                       'srch_ids': search_ids,
                       'prop_ids': prop_ids}) \
        .sort_values(['relevancies'], ascending=False)
    with open(f"output_{dataset}.txt", "w", newline='') as f:
        csv_writer = csv.DictWriter(f, ["srch_id", "prop_id"])
        csv_writer.writeheader()

        for srch_id, df2 in df.groupby('srch_ids'):
            for _, row in df2.iterrows():
                csv_writer.writerow({'srch_id': int(row['srch_ids']),
                                     'prop_id': int(row['prop_ids'])})


if __name__ == '__main__':
    train_and_save()
    predict_generate()
