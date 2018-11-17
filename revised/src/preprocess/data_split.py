from random import shuffle

def split_data(doc_id_set, train_portion = 0.8, test_portion = 0.1,
                num_val_subset = 1000, num_train_subset= 1000):

    # Make sure proportions are valid
    assert train_portion + test_portion <= 1

    validation_portion = round(1 - train_portion - test_portion,10)

    size = len(doc_id_set)

    # Turn our doc_ids into a list so we can shuffle it
    doc_id_list = list(doc_id_set)
    shuffle(doc_id_list)

    num_train = int(train_portion*size)
    num_validation = int(validation_portion*size)

    print(validation_portion)

    # Get the valildation set
    validation_ids = doc_id_list[:num_validation]
    rest = doc_id_list[num_validation:]

    # Get the train and test set from the rest
    train_ids = rest[:num_train]
    test_ids = rest[num_train:]

    shuffle(train_ids)


    if(len(validation_ids) < num_val_subset):
        # Return the whole validation set if num_val_subset is too big
        val_subset = validation_ids
    else:
        # Return a random subset wiht num_val_subset elemtents
        shuffle(validation_ids)
        val_subset = validation_ids[:num_val_subset]

    if(len(train_ids) < num_train_subset):
        # Return the whole validation set if num_test_subset is too big
        train_subset = test_ids
    else:
        # Return a random subset wiht num_test_subset elemtents
        shuffle(train_ids)
        train_subset = train_ids[:num_train_subset]



    return {"train": train_ids, "val": validation_ids,
            "test": test_ids, "val_subset": val_subset,
            "train_subset": train_subset }




print(split_data(set(range(10))))
