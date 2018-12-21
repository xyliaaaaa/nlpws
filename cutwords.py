import tensorflow as tf
import json


def load_settings():
    with open('char_id.json', 'r', encoding='utf-8') as f:
        char_id = json.load(fp=f)
    with open('id_char.json', 'r', encoding='utf-8') as f:
        id_char = json.load(fp=f)
        id_char = {int(id):char for id, char in id_char.items()}
    return char_id, id_char


def testsetprocess(test_txt):
    with open(test_txt, 'r', encoding='utf-8') as f:
        test_texts = f.readlines()
    test_X = []
    oov_id = {}
    for line in test_texts:
        ids = []
        for char in line:
            if char in char_id:
                ids.append(char_id[char])
            elif char in oov_id:
                ids.append(oov_id[char])
            else:
                oov_id[char] = len(char_id) + len(oov_id) + 1
                ids.append(oov_id[char])
        test_X.append(ids)
    return test_X, oov_id


def generate_text(test_X, test_Y):
    # tags = {"S": 1, "B": 2, "M": 3, "E": 4}
    texts = []
    for i in range(len(test_X)):
        line = []
        for j in range(len(test_X[i])):
            if test_Y[i][j] == 1 or test_Y[i][j] == 4:  # S or E
                line.append(f"{id_char[test_X[i][j]]}  ")
            elif test_Y[i][j] == 2 or test_Y[i][j] == 3:  # B or M
                line.append(f"{id_char[test_X[i][j]]}")
    texts.append(line)
    return texts


if __name__ == '__main__':
    char_id, id_char = load_settings()
    print(id_char)