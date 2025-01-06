import random

def randomRefSeq(sequence, seed):

    random.seed(seed)

    sequence_list = list(sequence)

    # 随机打乱顺序
    random.shuffle(sequence_list)

    # 将打乱后的列表重新组合成字符串
    shuffled_sequence = ''.join(sequence_list)

    return shuffled_sequence
