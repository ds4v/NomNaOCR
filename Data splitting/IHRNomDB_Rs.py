''' 
Implementation of the R-score algorithm for the IHRNomDB dataset:
- 𝑅_𝑠 = 𝑁_𝑑𝑖𝑠𝑡𝑖𝑛𝑐𝑡(𝑠) × max_𝑖^𝐷⁡〖𝑁_𝑖〗+ 𝑁_𝑠
- Implemnted by: Nguyen Duc Duy Anh (https://github.com/duyanh1909)
'''

from tqdm.notebook import tqdm
from collections import Counter


def print_intersection(val_data, train_data):
    """
    Calculate intersection.
    Params:
        val_data: the set of validation data.
        train_data: the set of train data.
    """
    total_chac_val = set(list(''.join(val_data)))
    total_chac_train = set(list(''.join(train_data)))

    word_val_in_train = len(list(filter(lambda x: True if x in total_chac_train else False, total_chac_val)))
    word_train_in_val = len(list(filter(lambda x: True if x in total_chac_val else False, total_chac_train)))

    intersection_val = word_val_in_train/len(total_chac_val) * 100
    intersection_train = word_train_in_val/len(total_chac_train) * 100

    print("Characters intersection train", intersection_val)
    print("Characters intersection val", intersection_train)


def frequence_in_D(dataset, char):
    """
    Count the number of appearances of a character in the dataset (D).
    Params:
        dataset (list): The dataset.
        char (str): The character to count.
    Returns:
        (int): The number of appearances of a character.
    """
    return ''.join(dataset).count(char)


def max_N(dataset):
    """
    Calculate 
    """
    max_score = 0
    vocab = dict(Counter(''.join(dataset)).most_common())

    for idx, s in tqdm(enumerate(dataset)):
        dataset_not_s = dataset[:idx] + dataset[idx:]
        list_s = list(s)
        vocab_not_s = vocab.copy()
        for word in list_s: vocab_not_s[word] -= 1
        
        sum_word_distinct = sum([vocab_not_s[word] for word in set(list_s)])
        if max_score < sum_word_distinct: max_score = sum_word_distinct
    return max_score


def calculate_r_scores(dataset):
    """
    Calculate R scores
    Params:
        dataset (list): The dataset.
    Returns:
        results (list): The list of Chinese character sequences with their r-scores.
    """
    vob = list(map(lambda elm: elm[1], dataset))
    results = []
    max_score = max_N(vob)
    vocab = dict(Counter(''.join(vob)).most_common())

    for idx, s in tqdm(enumerate(dataset)):
        list_s = list(s[1])
        vocab_not_s = vocab.copy()
        
        for word in list_s: vocab_not_s[word] -= 1
        sum_word_distinct = sum([vocab_not_s[word] for word in set(list_s)])
        sum_word = sum([vocab_not_s[word] for word in list_s])
        
        r = sum_word_distinct * max_score + sum_word
        results.append([s[0], s[1], r])
    return results