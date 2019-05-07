text = open('clean').read()
chars = sorted(list(set(text)))
char_size = len(chars)

"""
    create dictionary to link each char to an id, and vice versa
"""
char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))

print(chars)