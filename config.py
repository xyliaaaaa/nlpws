class Config:

    punctuations = ["”", "“", "？", "！", "。", "：", "，", "、", "（", "）", "《", "》", "——", "；", "‘", "’"]
    OOV = -1  # FIXME:暂时把testset中的OOV处理为-1,再看看有没有更好的处理方法
    tags = {"S": 1, "B": 2, "M": 3, "E": 4}
    dict_size = 0
    words_dict = {}
    seqlen = 32