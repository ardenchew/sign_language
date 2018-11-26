import numpy as np
from itertools import islice
from nltk.corpus import brown
import os.path


def split_every(n, iterable):
    # taken from stackoverflow
    i = iter(iterable)
    piece = ''.join(list(islice(i, n)))
    while piece:
        yield piece
        piece = ''.join(list(islice(i, n)))


def is_valid_ngram(s, n):
    return s.isalpha() and len(s) == n


def get_letter_index(letter):
    return ord(letter.lower()) - 97


def get_letter_from_index(index):
    return chr(index + 97)


def generate_ngram_probs(word_array, n=2):
    FILENAME = str(n) + "gram_probs.npy"
    if os.path.isfile(FILENAME):
        ngram_array = np.load(FILENAME)
    else:
        shape = [26] * n
        ngram_array = np.zeros(shape=shape, dtype=int)
    for word in word_array:
        for ngram in split_every(n, word.lower()):
            if is_valid_ngram(ngram, n):
                index_list = []
                for letter in ngram:
                    index_list.append(get_letter_index(letter))
                index_list = tuple(index_list)
                ngram_array[index_list] += 1
    np.save(FILENAME, ngram_array)
    return ngram_array


def get_next_letter_probs(prev_letters):
    MAX_N = 4
    genres = ['news', 'hobbies', 'romance']
    words = brown.words(categories=genres)
    row = None
    prev_letters = prev_letters[-(MAX_N - 1):]
    n = len(prev_letters) + 1
    index_list = []
    for letter in prev_letters:
        index_list.append(get_letter_index(letter))
    index_list = tuple(index_list)
    row = generate_ngram_probs(words, n=n)[index_list]
    row = row / np.sum(row)
    row = np.delete(row, get_letter_index('j'))
    row = np.delete(row, get_letter_index('z') - 1)
    return row