import pandas as pd
from config import Config
import json

class Preprocess:
    def __init__(self,train_txt):
        self.words_dict = {}
        self.id_dict = {}
        self.raw = self.load_train_data(train_txt)
        self.sorted_raw = sorted(self.raw,key = lambda x:len(x))
        self.tagged = self.to_tagged_texts(self.raw)
        self.train_X, self.train_Y = self.to_formatted_data(self.tagged)
        # self.sorted_dict = self.sort_dict()

    def load_train_data(self,train_txt):
        with open(train_txt,"r",encoding='utf-8') as f:
            raw_texts = f.readlines()
        return raw_texts

    def to_tagged_texts(self,raw_text):
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

    def to_formatted_data(self, tagged):
        train_X = []
        train_Y = []
        for sent in tagged:
            x = []
            y = []
            word_units = sent.split(" ")
            for unit in word_units:
                if len(unit.split("/")) == 2:
                    word, tag = unit.split("/")
                    # if word not in self.words_dict:
                    #     self.words_dict[word] = [len(self.words_dict)+1, 1]  # words_dict[word] = [id, counts]
                    #
                    # else:
                    #     self.words_dict[word][1] += 1

                    if word not in self.words_dict:
                        self.words_dict[word] = len(self.words_dict) + 1

                    x.append(self.words_dict[word])
                    y.append(Config.tags[tag])

            train_X.append(x)
            train_Y.append(y)



        return self.to_stdframe(train_X), self.to_stdframe(train_Y)

    def to_stdframe(self,list):
        df = pd.DataFrame(list)

        df = df.fillna(0)
        df = df.iloc[:, 0:Config.seqlen]
        return df.astype(int)

    # def sort_dict(self,words_num=None):
    #     sorted_dict = sorted(self.words_dict.items(),key=lambda item:item[1][1],reverse=True)
    #     return sorted_dict[:words_num]

    def id2char(self):
        # TODO: 试试字典是否能这么操作
        id_char = {value:key for key, value in self.words_dict.items()}
        self.id_dict = id_char


if __name__ == '__main__':
    data = Preprocess("train.txt")
    data.train_X.to_csv("train_X1.csv",index=False,header=False)
    data.train_Y.to_csv("train_Y1.csv",index=False,header=False)
    with open('words_id.json','w',encoding='utf-8') as f:
        json.dump(data.words_dict,f)
    with open('id_words.json','w',encoding='utf-8') as f:
        json.dump(data.id_dict,f)

