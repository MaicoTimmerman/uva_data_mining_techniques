import pandas as pd
from random import shuffle

def tinify_set(input_file, output_file, n_searches):
    raw_dataset = pd.read_csv(input_file)
    search_ids = raw_dataset['srch_id'].unique()
    shuffle(search_ids)
    selected_search_ids = search_ids[:n_searches]
    subselection = raw_dataset['srch_id'].isin(selected_search_ids)
    subsampled_dataset = raw_dataset[subselection]
    subsampled_dataset.to_csv(path_or_buf=output_file, index=False)
    print(output_file, selected_search_ids)


# tinify_set('training_set_VU_DM.csv', 'tiny_train.csv', 1000)
# tinify_set('test_set_VU_DM.csv', 'tiny_test.csv', 1000)
tinify_set('training_set_VU_DM.csv', 'tenth_train.csv', 35000)
tinify_set('test_set_VU_DM.csv', 'tenth_valid.csv', 7000)
tinify_set('test_set_VU_DM.csv', 'tenth_test.csv', 7000)