import pandas as pd

def prop_id_overlap (file1, file2):
    prop_ids_1 = pd.read_csv('../data/'+file1, dtype={'prop_id':int})['prop_id'].unique()
    prop_ids_2 = pd.read_csv('../data/'+file2, dtype={'prop_id':int})['prop_id']
    match_count = sum(prop_ids_2.isin(prop_ids_1))
    return match_count, len(prop_ids_1), len(prop_ids_2)

def print_overlap (set1, set2):
    matches, len1, len2 = prop_id_overlap(set1, set2)
    print(f"{matches} out of {len2} prop_ids from {set2} are present in {set1}. That is an overlap of {100.0*matches/len2:.2f}%.")

print_overlap('tiny_train.csv', 'tiny_test.csv')
print_overlap('training_set_VU_DM.csv', 'test_set_VU_DM.csv')

# 11562 out of 24754 prop_ids from tiny_test.csv are present in tiny_train.csv. That is an overlap of 46.71%.
# 4945334 out of 4959183 prop_ids from test_set_VU_DM.csv are present in training_set_VU_DM.csv. That is an overlap of 99.72%.
