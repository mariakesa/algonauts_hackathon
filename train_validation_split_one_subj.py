import os
from sklearn.model_selection import train_test_split

subj_string = 'subj07/'
pre_path = '/home/maria/Desktop/algonauts_data/'


def generate_train_test_split_subj(pre_path, subj_string):
    files = os.listdir(pre_path+subj_string+'training_split/training_images/')
    ind_dct = {}
    i = 0
    for f in sorted(files):
        ind_dct[i] = f
        i += 1
    train_inds, val_inds = train_test_split(
        range(0, len(files)), test_size=0.25, random_state=71)
    with open(subj_string[:-1]+".train", "w") as f:
        f.write("\n".join([subj_string+'training_split/training_images/' +
                           str(ind_dct[i]) for i in train_inds]))
    with open(subj_string[:-1]+".val", "w") as f:
        f.write("\n".join([subj_string+'training_split/training_images/' +
                           str(ind_dct[i]) for i in val_inds]))


generate_train_test_split_subj(pre_path, subj_string)
