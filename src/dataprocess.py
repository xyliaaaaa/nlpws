# -*- coding: utf-8 -*-
import numpy as np
from config import Config
import json


def to_tagged_texts(raw_text):
    '''
    训练集去掉空格，打上标签
    采用4标签。tags = {"S": 1, "B": 2, "M": 3, "E": 4}
    :param raw_text: list.带空格的训练集文本
    :return: list.每个字符后打上标签的文本
    '''
    tagged = []
    for line in raw_text:
        output_line = ""
        for i in range(len(line)):
            if line[i] not in [" ", "\r", "\n"]:
                # 句子长度大于60，则按标点符号分割
                if len(output_line) > 60 and line[i] in Config.punctuations:
                    output_line += f"{line[i]}/S "
                    tagged.append(output_line)
                    # tagged.append(output_line)
                    output_line = ""

                # 否则按换行符分割
                elif i == 0:  # 行首
                    if line[i + 1] == ' ':
                        output_line += f"{line[i]}/S "
                    else:
                        output_line += f"{line[i]}/B "
                elif line[i - 1] == ' ':  # 不在行首且前面是空格
                    if line[i + 1] in [' ', '\r','\n']:  # 行尾或者后面是空格
                        output_line += f"{line[i]}/S "
                    else:
                        output_line += f"{line[i]}/B "
                elif i == len(line) - 1 or line[i + 1] == ' ':  # 前面不是空格，行尾或者后面是空格
                    output_line += f"{line[i]}/E "
                else:
                    output_line += f"{line[i]}/M "
        if output_line != "":
            tagged.append(output_line)
    return tagged


def split(tagged):
    '''
    分离出标签和训练数据，生成字符字典
    :param tagged: list. 带标签的训练文本
    :return: 2darray：train_X and train_Y. dictionary: char_id
    '''
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
                if char not in char_id:
                    char_id[char] = len(char_id) + 1  # 索引值从1开始

                x.append(char_id[char])
                y.append(Config.tags[tag])

        train_X.append(x)
        train_Y.append(y)
    return np.array(train_X), np.array(train_Y), char_id


# def id2char(char_id):
#     id_char = {value:key for key, value in char_id.items()}
#     return id_char


def trainsetprocess(train_txt):
    '''
    训练集预处理
    :param train_txt: 训练集文件路径
    :return: 整数型2darray：训练集数据和标签。dictionary：字符到索引值的字典
    '''
    with open(train_txt, 'r', encoding='utf-8') as f:
        raw_texts = f.readlines()

    # 根据空格打上标签
    tagged = to_tagged_texts(raw_texts)
    with open("../data/tagged.txt",'w',encoding='utf-8') as f:
        for line in tagged:
            f.write(f"{line}\n")
    train_X, train_Y, char_id = split(tagged)
    # id_char = id2char(char_id)

    # 保存数据
    np.save("../data/train_X.npy",train_X)
    np.save("../data/train_Y.npy",train_Y)
    print("Training Data Saved")
    with open('../data/char_id.json', 'w', encoding='utf-8') as f:
        json.dump(char_id, f, ensure_ascii=False)
    # with open('../data/id_char.json', 'w', encoding='utf-8') as f:
    #     json.dump(id_char, f, ensure_ascii=False)


def testsetprocess(test_txt):
    '''
    测试集预处理
    :param test_txt: 测试集文本
    :return: 切碎的文本和转换为索引值的array
    '''
    with open(test_txt, 'r', encoding='utf-8') as f:
        text = f.readlines()

    testtext = []
    for line in text:
        # 长度小于50则按换行符切割
        if len(line) < 50:
            testtext.append(line)
        # 否则按标点符号切割
        else:
            sent = ""
            for char in line:
                sent += char
                if char in line[:-5] and char in Config.punctuations:
                    sent += "\t"  # '\t'为切割标记
            testtext.append(sent)
    short_text = []
    for str in testtext:
            short_text.extend(str.split("\t"))
    test_X = np.array(test_label(short_text))
    short_text = np.array(short_text)

    # 保存数据
    np.save('../data/test_X.npy', test_X)
    np.save('../data/test_text.npy', short_text)
    print('Test Data Saved')


def test_label(testtext):
    with open('../data/char_id.json', 'r', encoding='utf-8') as f:
        char_id = json.load(fp=f)  # 载入字典
    test_X = [[char_id.get(char) if char in char_id else 5999 for char in line] for line in testtext]
    return test_X


def generate_text(test_text, test_Y, outputfile):
    # tags = {"S": 1, "B": 2, "M": 3, "E": 4}
    text = []
    for i in range(len(test_text)):
        line = ""
        for j in range(len(test_text[i])):
            # add a strong rule: cut when come across punctuations
            if test_text[i][j] in Config.punctuations:
                line += f'{test_text[i][j]} '
            elif test_Y[i][j] in [1,4,0]:
                line += f'{test_text[i][j]} ' # cut
            else:
                # test_Y[i][j] in [2,3]:
                line += f'{test_text[i][j]}' # do not cut

        text.append(line)
    # 保存预测结果
    with open(outputfile, 'w', encoding='utf-8') as f:
        f.writelines(text)


#
# def merge(test_text,test_Y):
#     text = []
#     for i in range(len(test_text)):
#         line = ""
#         for j in range(len(test_text[i])):
#             line += f'{test_text[i][j]}{test_Y[i][j]}'
#         text.append(line)
#     return text

if __name__ == '__main__':
    # trainsetprocess("../data/train.txt")
    # testsetprocess("../data/test.txt")
    test_text = np.load('../data/test_text.npy')
    test_Y = np.load('../data/test_Y.npy')
    generate_text(test_text, test_Y, '../data/test_output.txt')

