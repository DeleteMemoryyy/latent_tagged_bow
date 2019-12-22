from nltk.tokenize import word_tokenize
from config import config


class Dataloader:
    def __init__(self):
        self.path = config.dataset_path
        self.test_index_path = config.dataset_test_index_path
        self.sentences = []
        self.tokens = []
        self.test_index = []
        self.train_tokens = []
        self.test_tokens = []

    def load(self):
        with open(self.path, encoding='utf-8') as fd:
            lines = fd.readlines()
        for line in lines:
            s1, s2 = line[:-1].split('\t')
            self.sentences.append((s1, s2))
            self.tokens.append((word_tokenize(s1), word_tokenize(s2)))
        self.all_num = len(self.sentences)

        with open(self.test_index_path, encoding='utf-8') as fd:
            lines = fd.readlines()
        for line in lines:
            index = int(line[:-1])
            self.test_index.append(index)
        self.test_num = len(self.test_index)

        print(len(self.tokens))

    def spilt_train_test(self):
        flag_table = [True] * self.all_num
        for index in self.test_index:
            if index >= self.all_num:
                continue
            flag_table[index] = False

        for index, token in enumerate(self.tokens):
            if flag_table[index]:
                self.train_tokens.append(token)
            else:
                self.test_tokens.append(token)
