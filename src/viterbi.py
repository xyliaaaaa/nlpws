

def viterbi(nodes):
    tags = {"S": 1, "B": 2, "M": 3, "E": 4}
    states = tags
    trans_pro =  {'BE': 0.5, 'BM': 0.5, 'EB': 0.5, 'ES': 0.5, 'ME': 0.5, 'MM': 0.5, 'SB': 0.5, 'SS': 0.5}
    path = {s:[] for s in ['S','B']}
    s_pro = {'S': 0.5, 'B': 0.5, 'M': 0, 'E': 0}


    curr_pro = {}
    for s in states:
        curr_pro[s] = s_pro[s]
    for i in range(1, len(nodes)):
        last_pro = curr_pro
        curr_pro = {}
        for curr_state in states:
            max_pro, last_sta = max(((last_pro[last_state]*trans_pro[last_state + curr_state],last_state) for last_state in states))
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)

    # 寻找概率最大路径
    max_pro = -1
    max_path = None
    for s in states:
        path[s].append(s)
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
    return max_path
# def viterbi(nodes):
#     trans = {'be': 0.5, 'bm': 0.5, 'eb': 0.5, 'es': 0.5, 'me': 0.5, 'mm': 0.5, 'sb': 0.5, 'ss': 0.5}
#     paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
#     for l in range(1, len(nodes)):
#         paths_ = paths.copy()
#         paths = {}
#         for i in nodes[l].keys():
#             nows = {}
#             for j in paths_.keys():
#                 if j[-1] + i in trans.keys():
#                     nows[j + i] = paths_[j] + nodes[l][i] + trans[j[-1] + i]
#             nows = sorted(nows.items(), key=lambda x: x[1], reverse=True)
#             paths[nows[0][0]] = nows[0][1]
#
#     paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
#     return paths[0][0]
#
