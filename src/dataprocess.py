import pandas as pd
from config import Config
import json


def to_tagged_texts(raw_text):
    # punctuations = ["”", "“", "？", "！", "。", "：", "，", "、", "（", "）", "《", "》", "——", "；", "‘", "’"]
    tagged = []
    for line in raw_text:
        output_line = ""
        for i in range(len(line)):
            if line[i] not in [" ", "\r", "\n"]:
                # if line[i] not in [" ", "\r", "\n"] and line[i] not in punctuations:
                # if line[i] in punctuations:
                #     output_line += f"{line[i]}/P "
                if i == 0:
                    if line[i + 1] == ' ':
                        output_line += f"{line[i]}/S "
                    else:
                        output_line += f"{line[i]}/B "
                # 不在行首
                # 前面是空格
                elif line[i - 1] == ' ':
                    if i == len(line) - 1 or line[i + 1] == ' ':  # 行尾或者后面是空格
                        output_line += f"{line[i]}/S "
                    else:
                        output_line += f"{line[i]}/B "
                # 前面不是空格，行尾或者后面是空格
                elif i == len(line) - 1 or line[i + 1] == ' ':
                    output_line += f"{line[i]}/E "
                else:
                    output_line += f"{line[i]}/M "
        tagged.append(f"{output_line}\r\n")
    return tagged


def to_stdframe(list):
    df = pd.DataFrame(list)

    df = df.fillna(0)
    df = df.iloc[:, 0:Config.seqlen]
    return df.astype(int)

def to_formatted_data(tagged):
    train_X = []
    train_Y = []
    char_id = {}
    for sent in tagged:
        x = []
        y = []
        word_units = sent.split(" ")
        for unit in word_units:
            if len(unit.split("/")) == 2:
                char, tag = unit.split("/")
                # if word not in self.words_dict:
                #     self.words_dict[word] = [len(self.words_dict)+1, 1]  # words_dict[word] = [id, counts]
                #
                # else:
                #     self.words_dict[word][1] += 1

                if char not in char_id:
                    char_id[char] = len(char_id) + 1

                x.append(char_id[char])
                y.append(Config.tags[tag])

        train_X.append(x)
        train_Y.append(y)
    return to_stdframe(train_X), to_stdframe(train_Y),char_id


def id2char(char_id):
    id_char = {}
    id_char = {value:key for key, value in char_id.items()}
    return id_char


def trainsetprocess(train_txt):
    with open(train_txt, "r", encoding='utf-8') as f:
        raw_texts = f.readlines()
    tagged = to_tagged_texts(raw_texts)
    train_X, train_Y, char_id = to_formatted_data(tagged)
    id_char = id2char(char_id)
    train_X.to_csv("train_X1.csv", index=False, header=False)
    train_Y.to_csv("train_Y1.csv", index=False, header=False)
    with open('char_id.json', 'w', encoding='utf-8') as f:
        json.dump(char_id, f)
    with open('id_char.json', 'w', encoding='utf-8') as f:
        # FIXME:文件名改为id_word.json
        json.dump(id_char, f)



if __name__ == '__main__':
    trainsetprocess("train.txt")
