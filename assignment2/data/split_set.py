from random import shuffle

import pandas as pd


def split_set(input_file, output_file_1, output_file_2, ratio):
    raw_dataset = pd.read_csv(input_file)
    search_ids = raw_dataset['srch_id'].unique()
    shuffle(search_ids)
    n_samples = (int)(len(search_ids)*(1-ratio))
    set1_ids = search_ids[:n_samples]
    set2_ids = search_ids[n_samples:]
    set1_selection = raw_dataset['srch_id'].isin(set1_ids)
    set2_selection = raw_dataset['srch_id'].isin(set2_ids)
    set1 = raw_dataset[set1_selection]
    set2 = raw_dataset[set2_selection]
    set1.to_csv(path_or_buf=output_file_1, index=False)
    set2.to_csv(path_or_buf=output_file_2, index=False)


split_set('training_set_VU_DM.csv', '90_train.csv', 'temp.csv', 0.1)
split_set('temp.csv', '90_valid.csv', '90_test.csv', 0.5)
