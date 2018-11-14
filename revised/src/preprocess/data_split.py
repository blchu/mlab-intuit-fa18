from random import shuffle

def split_data(doc_id_set, train_portion = 0.8, test_portion = 0.1,
                num_val_subset = 1000, num_test_subset = 1000):

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

    if(len(test_ids) < num_test_subset):
        # Return the whole validation set if num_test_subset is too big
        test_subset = test_ids
    else:
        # Return a random subset wiht num_test_subset elemtents
        shuffle(test_ids)
        test_subset = test_ids[:num_test_subset]



    return {"train": set(train_ids), "val": set(validation_ids),
            "test": set(test_ids), "val_subset":set(val_subset) ,
            "test_subset": set(test_subset) }




print(split_data(set(range(10))))
