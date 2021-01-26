def create_vocabulary(words):
    """
    Returns a word2idx and a idx2word dict
    """
    word2idx = dict()
    idx2word = dict()
    for i, word in enumerate(sorted(set(words))):
        word2idx[word] = i
        idx2word[i] = word
    return word2idx, idx2word
