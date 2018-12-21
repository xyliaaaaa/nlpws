import pandas as pd
from config import Config
from keras.preprocessing.text import Tokenizer

class dataFormat:
    '''
    文本数据实例及对它的几个操作
    '''
    dict_size = 0
    words_dict = {}

    def __init__(self,text):
        self.filename = text


    def to_dataframe(self,text,mode):
        '''
        convert test text to frame
        :param testtext:
        :return:
        '''
        if mode == "test":
            test_X = []
            for sent in text:
                x = []
                for word in sent:
                    if word in self.words_dict:
                        x.append(self.words_dict[word])
                    else:
                        x.append(Config.OOV)
                test_X.append(x)
            return self.to_stdframe(test_X)

        elif mode == "train":
            train_X = []
            train_Y = []
            for sent in text:
                x = []
                y = []
                word_units = sent.split(" ")
                for unit in word_units:
                    if len(unit.split("/")) == 2:
                        word, tag = unit.split("/")
                        # if word not in self.words_dict:
                        #     dataFormat.dict_size += 1
                        #     dataFormat.words_dict[word] = dataFormat.dict_size
                        # x.append(dataFormat.words_dict[word])
                        y.append(Config.tags[tag])
                # train_X.append(x)
                train_Y.append(y)

            tokenizer = Tokenizer(num_words=999,filters='/SBEMP',split=' ',char_level=True)
            tokenizer.fit_on_texts(text)
            train_X = tokenizer.texts_to_sequences(text)
            return self.to_stdframe(train_X), self.to_stdframe(train_Y)
        else:
            print("wrong mode name")
            return -1


    def to_tagged_txt(self, outputfile):
        with open(self.filename, 'r', encoding='utf-8') as f:
            txt = f.readlines()
        tagged_text = self.trainset_tagging(txt)

        with open(outputfile, 'w', encoding='utf-8') as f:
            for line in tagged_text:
                f.write(line)


    def trainset_tagging(self,txt):

        # punctuations = ["”", "“", "？", "！", "。", "：", "，", "、", "（", "）", "《", "》", "——", "；", "‘", "’"]
        tagged = []
        for line in txt:
            output_line = ""
            for i in range(len(line)):
                if line[i] not in [" ", "\r", "\n"]:
                    if line[i] in Config.punctuations:
                        output_line += f"{line[i]}/P "
                    elif i == 0:
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


    def to_stdframe(self,list):
        df = pd.DataFrame(list)
        df = df.iloc[:, 0:Config.seqlen]
        return df.fillna(0)

    def read_tagged_txt(self,tagged):
        with open(tagged, 'r', encoding='utf-8') as f:
            text = f.readlines()

        X, Y = data_train.to_dataframe(text, "train")
        return X,Y


if __name__ == '__main__':
    # TODO:这个地方把函数再封装好
    # train = "train.txt"
    # tagged = "train_tagged.txt"
    #
    # data_train = dataFormat(train)
    # # data_train.to_tagged_txt(tagged)
    # # print(dataFormat.dict_size)
    #
    # X,Y = data_train.read_tagged_txt(tagged)
    # print(dataFormat.dict_size)
    #
    # X.to_csv("train_X11.csv")
    # Y.to_csv("train_Y11.csv")

    print(Tokenizer.word_counts)
