import pandas as pd

def balanced_set(input_file, output_file):
    raw_dataset = pd.read_csv(input_file)
    balanced_dataset = raw_dataset.drop_duplicates(subset=['srch_id', 'click_bool', 'booking_bool'])
    balanced_dataset.to_csv(path_or_buf=output_file, index=False)


balanced_set('training_set_VU_DM.csv', 'balanced_train.csv')