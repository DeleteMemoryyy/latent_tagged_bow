import pickle
from collections import Counter

import nltk
import numpy as np
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm


def quora_read(file_path, bleu_baseline=False):
    """Read the quora dataset"""
    print("Reading quora raw data .. ")
    print("  data path: %s" % file_path)
    with open(file_path, encoding='utf-8') as fd:
        lines = fd.readlines()
    sentence_sets = []
    for l in tqdm(lines):
        p0, p1 = l[:-1].lower().split("\t")
        sentence_sets.append([word_tokenize(p0), word_tokenize(p1)])

    if (bleu_baseline):
        print("calculating bleu ... ")
        hypothesis = [s[0] for s in sentence_sets]
        references = [s[1:] for s in sentence_sets]
        bleu = corpus_bleu(references, hypothesis)
        print("bleu on the training set: %.4f" % bleu)
    return sentence_sets


def build_batch_bow_seq2seq(data_batch,
                            len_batch,
                            stop_words,
                            max_enc_bow,
                            max_dec_bow,
                            pad_id):
    """Build a training batch for the bow seq2seq model"""
    enc_inputs = []
    pos_inputs = []
    ner_inputs = []
    enc_lens = []
    enc_targets = []
    dec_bow = []
    dec_bow_len = []
    dec_inputs = []
    dec_targets = []
    dec_lens = []

    pos_batch = [j for i, j, k in data_batch]
    ner_batch = [k for i, j, k in data_batch]
    data_batch = [i for i, j, k in data_batch]

    def _pad(s_set, max_len, pad_id):
        s_set = list(s_set)[: max_len]
        for i in range(max_len - len(s_set)): s_set.append(pad_id)
        return s_set

    for st, slen, pos, ner in zip(data_batch, len_batch, pos_batch, ner_batch):
        para_bow = set()
        for s in st: para_bow |= set(s)
        para_bow -= stop_words

        num_para = len(st)

        for i in range(num_para):
            j = (i + 1) % num_para
            inp = st[i][: -1]
            d_in = st[j][: -1]
            d_out = st[j][1:]
            len_inp = slen[i]
            len_out = slen[j]

            enc_inputs.append(inp)
            pos_inputs.append(pos[i][: -1])
            ner_inputs.append(ner[i][: -1])
            enc_lens.append(len_inp)

            d_bow = set(d_in) - stop_words
            d_bow_len = len(d_bow)
            d_bow = _pad(d_bow, max_dec_bow, pad_id)

            e_bow = []
            for k in range(num_para):
                if (k == i):
                    continue
                    # if(source_bow == False):
                    #   continue
                e_bow.extend(st[k][: -1])
            e_bow = set(e_bow) - stop_words

            # method 1: pad bow
            # do not predict source words
            e_bow = _pad(e_bow, max_enc_bow, pad_id)

            enc_targets.append(e_bow)
            dec_bow.append(d_bow)
            dec_bow_len.append(d_bow_len)
            dec_inputs.append(d_in)
            dec_targets.append(d_out)
            dec_lens.append(len_out)

    batch_dict = {"enc_inputs": np.array(enc_inputs),
                  "enc_lens": np.array(enc_lens),
                  "enc_targets": np.array(enc_targets),
                  "dec_bow": np.array(dec_bow),
                  "dec_bow_len": np.array(dec_bow_len),
                  "dec_inputs": np.array(dec_inputs),
                  "dec_targets": np.array(dec_targets),
                  "dec_lens": np.array(dec_lens),
                  "pos_inputs": np.array(pos_inputs),
                  "ner_inputs": np.array(ner_inputs)}
    return batch_dict


def build_batch_bow_seq2seq_eval(
        data_batch, len_batch, stop_words, max_enc_bow, max_dec_bow, pad_id,
        single_ref=False):
    """Build evaluation batch, basically the same as the seq2seq setting"""
    enc_inputs = []
    pos_inputs = []
    ner_inputs = []
    enc_lens = []
    references = []
    dec_golden_bow = []
    dec_bow = []
    dec_bow_len = []

    pos_batch = [j for i, j, k in data_batch]
    ner_batch = [k for i, j, k in data_batch]
    data_batch = [i for i, j, k in data_batch]

    def _pad(s_set, max_len, pad_id):
        s_set = list(s_set)[: max_len]
        for i in range(max_len - len(s_set)): s_set.append(pad_id)
        return s_set

    def _pad_golden(s_set, max_len):
        s_set_ = list(s_set)
        s_set = list(s_set)[: max_len]
        for i in range(max_len - len(s_set)):
            s_set.append(np.random.choice(s_set_))
        return s_set

    for st, slen, pos, ner in zip(data_batch, len_batch, pos_batch, ner_batch):
        inp = st[0][: -1]
        pos_inputs.append(pos[0][: -1])
        ner_inputs.append(ner[0][: -1])
        len_inp = slen[0]
        if (single_ref):
            ref = [s_[1: s_len_] for s_, s_len_ in zip(st[-1:], slen[-1:])]
        else:
            ref = [s_[1: s_len_] for s_, s_len_ in zip(st[1:], slen[1:])]

        golden_bow = []
        for r in ref: golden_bow.extend(r)
        golden_bow = set(golden_bow)
        golden_bow = golden_bow - stop_words
        golden_bow = _pad_golden(golden_bow, max_enc_bow)

        d_in = st[1][: -1]
        d_bow = set(d_in) - stop_words
        d_bow_len = len(d_bow)
        d_bow = _pad(d_bow, max_dec_bow, pad_id)
        dec_bow.append(d_bow)
        dec_bow_len.append(d_bow_len)

        enc_inputs.append(inp)
        enc_lens.append(len_inp)
        references.append(ref)
        dec_golden_bow.append(golden_bow)

    batch_dict = {"enc_inputs": np.array(enc_inputs),
                  "enc_lens": np.array(enc_lens),
                  "golden_bow": np.array(dec_golden_bow),
                  "dec_bow": np.array(dec_bow),
                  "dec_bow_len": np.array(dec_bow_len),
                  "references": references,
                  "pos_inputs": np.array(pos_inputs),
                  "ner_inputs": np.array(ner_inputs)}
    return batch_dict


def train_dev_split(dataset_name, train_sets, train_index_file, dev_index_file):
    """Suffle the dataset and split the training set"""
    print("Splitting training and dev set ... ")

    if (dataset_name == "quora"):
        with open(train_index_file, encoding='utf-8') as fd:
            train_index = set([int(l[:-1]) for l in fd.readlines()])

        with open(dev_index_file, encoding='utf-8') as fd:
            dev_index = set([int(l[:-1]) for l in fd.readlines()])

        train, dev = [], []
        for i in range(len(train_sets)):
            if (i in train_index):
                train.append(train_sets[i])
            elif (i in dev_index):
                dev.append(train_sets[i])

    print("Size of training set: %d" % len(train))
    print("Size of test set: %d" % len(dev))
    return train, dev


class Dataset(object):
    """The dataset class, read the raw data, process into intermediate
    representation, and load the intermediate as batcher"""

    def __init__(self, config):
        """Initialize the dataset configuration"""
        self.dataset = config.dataset
        self.dataset_path = config.dataset_path[self.dataset]
        self.train_index_path = config.train_index_path
        self.dev_index_path = config.dev_index_path
        self.read_from_pickle = config.read_from_pickle
        self.word2id_path = config.word2id_path
        self.id2word_path = config.id2word_path
        self.pos2id_path = config.pos2id_path
        self.id2pos_path = config.id2pos_path
        self.ner2id_path = config.ner2id_path
        self.id2ner_path = config.id2ner_path
        self.train_sentences_path = config.train_sentences_path
        self.train_lens_path = config.train_lens_path
        self.dev_sentences_path = config.dev_sentences_path
        self.dev_lens_path = config.dev_lens_path
        self.max_sent_len = config.max_sent_len[self.dataset]
        self.max_enc_bow = config.max_enc_bow
        self.max_dec_bow = config.max_dec_bow
        self.single_ref = config.single_ref

        self.stop_words = set(stopwords.words('english'))

        self.word2id = None
        self.id2word = None
        self.pad_id = -1
        self.start_id = -1
        self.eos_id = -1
        self.unk_id = -1

        self._dataset = {"train": None, "dev": None, "test": None}
        self._sent_lens = {"train": None, "dev": None, "test": None}
        self._ptr = {"train": 0, "dev": 0, "test": 0}
        return

    @property
    def vocab_size(self):
        return len(self.word2id)

    def dataset_size(self, setname):
        return len(self._dataset[setname])

    def num_batches(self, batch_size, setname):
        setsize = self.dataset_size(setname)
        num_batches = setsize // batch_size + 1
        return num_batches

    def build(self):
        """Build the dataset to intermediate representation

        The data processing pipeline:
        * read raw file
        * calculate corpus statistics
        * split training and validation
        * build vocabulary
        * normalize the text
        """
        # read training sentences
        if self.read_from_pickle:
            with open(self.word2id_path, "rb") as word2id_file:
                word2id = pickle.load(word2id_file)
            with open(self.id2word_path, "rb") as id2word_file:
                id2word = pickle.load(id2word_file)
            with open(self.pos2id_path, "rb") as pos2id_file:
                pos2id = pickle.load(pos2id_file)
            with open(self.id2pos_path, "rb") as id2pos_file:
                id2pos = pickle.load(id2pos_file)
            with open(self.ner2id_path, "rb") as ner2id_file:
                ner2id = pickle.load(ner2id_file)
            with open(self.id2ner_path, "rb") as id2ner_file:
                id2ner = pickle.load(id2ner_file)

            with open(self.train_sentences_path, "rb") as train_sentences_file:
                train_sentences = pickle.load(train_sentences_file)
            with open(self.train_lens_path, "rb") as train_lens_file:
                train_lens = pickle.load(train_lens_file)
            with open(self.dev_sentences_path, "rb") as dev_sentences_file:
                dev_sentences = pickle.load(dev_sentences_file)
            with open(self.dev_lens_path, "rb") as dev_lens_file:
                dev_lens = pickle.load(dev_lens_file)


        else:
            train_sentences = quora_read(self.dataset_path["train"])

            train_sentences, dev_sentences = train_dev_split(
                self.dataset, train_sentences, self.train_index_path, self.dev_index_path)

            word2id, id2word, pos2id, id2pos, ner2id, id2ner = get_vocab(train_sentences)
            with open(self.word2id_path, "wb") as word2id_file:
                pickle.dump(word2id, word2id_file)
            with open(self.id2word_path, "wb") as id2word_file:
                pickle.dump(id2word, id2word_file)
            with open(self.pos2id_path, "wb") as pos2id_file:
                pickle.dump(pos2id, pos2id_file)
            with open(self.id2pos_path, "wb") as id2pos_file:
                pickle.dump(id2pos, id2pos_file)
            with open(self.ner2id_path, "wb") as ner2id_file:
                pickle.dump(ner2id, ner2id_file)
            with open(self.id2ner_path, "wb") as id2ner_file:
                pickle.dump(id2ner, id2ner_file)

            train_sentences, train_lens = normalize(
                train_sentences, word2id, pos2id, ner2id, self.max_sent_len)
            dev_sentences, dev_lens = normalize(
                dev_sentences, word2id, pos2id, ner2id, self.max_sent_len)
            with open(self.train_sentences_path, "wb") as train_sentences_file:
                pickle.dump(train_sentences, train_sentences_file)
            with open(self.train_lens_path, "wb") as train_lens_file:
                pickle.dump(train_lens, train_lens_file)
            with open(self.dev_sentences_path, "wb") as dev_sentences_file:
                pickle.dump(dev_sentences, dev_sentences_file)
            with open(self.dev_lens_path, "wb") as dev_lens_file:
                pickle.dump(dev_lens, dev_lens_file)

        self.word2id = word2id
        self.id2word = id2word
        self.pos2id = pos2id
        self.id2pos = id2pos
        self.ner2id = ner2id
        self.id2ner = id2ner
        self.start_id = word2id["_GOO"]
        self.eos_id = word2id["_EOS"]
        self.unk_id = word2id["_UNK"]
        self.pad_id = word2id["_PAD"]
        self.stop_words = set([word2id[w] if (w in word2id) else self.pad_id for w in self.stop_words])
        self.stop_words |= {self.start_id, self.eos_id, self.unk_id, self.pad_id}

        self._dataset["train"] = train_sentences
        self._dataset["test"] = dev_sentences
        # self._dataset["test"] = test_sentences
        self._sent_lens["train"] = train_lens
        self._sent_lens["test"] = dev_lens
        # self._sent_lens["test"] = test_lens
        return

    def next_batch(self, setname, batch_size):
        """Get next data batch

        Args:
          setname: a string, "train", "valid", or "test"
          batch_size: the size of the batch, an integer
          model_name: the name of the model, a string, different model use different
            data representations
        """
        ptr = self._ptr[setname]
        data_batch = self._dataset[setname][ptr: ptr + batch_size]
        len_batch = self._sent_lens[setname][ptr: ptr + batch_size]

        if (setname == "train"):
            batch_dict = build_batch_bow_seq2seq(data_batch, len_batch,
                                                 self.stop_words, self.max_enc_bow, self.max_dec_bow, self.pad_id)
        else:  # evaluation
            batch_dict = build_batch_bow_seq2seq_eval(data_batch, len_batch,
                                                      self.stop_words, self.max_enc_bow, self.max_dec_bow,
                                                      self.pad_id,
                                                      self.single_ref)

        ptr += batch_size
        if (ptr == self.dataset_size(setname)):
            ptr = 0
        if (ptr + batch_size > self.dataset_size(setname)):
            ptr = self.dataset_size(setname) - batch_size
        self._ptr[setname] = ptr
        return batch_dict

    def decode_sent(self, sent, sent_len=-1, prob=None):
        """Decode the sentence index"""
        s_out = []
        is_break = False
        for wi, wid in enumerate(sent[:sent_len]):
            if (is_break): break
            w = self.id2word[wid]
            if (w == "_EOS"):
                is_break = True
            s_out.append(w)
            if (prob is not None): s_out.append("(%.3f) " % prob[wi])
        return " ".join(s_out)

    def decode_neighbor(self, sent, neighbor_ind, neighbor_prob, sent_len=-1):
        """Decode the predicted sentence neighbot"""
        s_out = ""
        is_break = False
        for wid, nb, np in zip(
                sent[: sent_len], neighbor_ind[: sent_len], neighbor_prob[: sent_len]):
            if (is_break): break
            w = self.id2word[wid]
            s_out += "%s: " % w
            for b, p in zip(nb, np):
                bw = self.id2word[b]
                s_out += "%s(%.4f), " % (bw, p)
            s_out += "\n"

            if (w == "_EOS"):
                is_break = True
        return s_out

    def print_predict(self, output_dict, batch_dict, fd=None):
        """Print out the prediction"""
        self.print_predict_latent_bow(output_dict, batch_dict, fd)
        return

    def print_predict_latent_bow(self, output_dict, batch_dict, fd=None):
        """Print the prediction, latent bow model

        Things to print
        * The input sentence,
        * The predicted bow and their probabilities
        * The sampled bow and their probabilities
        * The predicted sentences
        * The references
        """
        if (fd == None):
            print_range = 5
        else:
            print_range = len(output_dict["dec_predict"])
        for i in range(print_range):
            out = "inputs:\n"
            out += "    " + self.decode_sent(batch_dict["enc_inputs"][i]) + "\n"
            out += "input neighbors:\n"
            out += self.decode_neighbor(batch_dict["enc_inputs"][i],
                                        output_dict["seq_neighbor_ind"][i], output_dict["seq_neighbor_prob"][i])
            out += "enc_output_bow:\n"
            out += "    " + self.decode_sent(output_dict["bow_pred_ind"][i]) + "\n"
            out += "enc_sampled_memory:\n"
            out += "    " + self.decode_sent(output_dict["sample_memory_ind"][i]) + "\n"
            out += "dec_outputs:\n"
            out += "    " + self.decode_sent(output_dict["dec_predict"][i]) + "\n"
            out += "references:\n"
            for r in batch_dict["references"][i]:
                out += "    " + self.decode_sent(r) + "\n"
            if (fd != None):
                fd.write(out + "\n")
            else:
                print(out)
        return


# nlp tools
def normalize(sentence_sets, word2id, pos2id, ner2id, max_sent_len):
    """Normalize the sentences by the following procedure
    - word to index
    - add unk
    - pad/ cut the sentence length
    - record the sentence length

    Args:
      sentence_sets: the set of sentence paraphrase, a list of sentence list
      word2id: word index, a dictionary
      max_sent_len: maximum sentence length, a integer
    """
    sent_sets = []
    sent_len_sets = []
    max_sent_len = max_sent_len + 1

    pos_sets = []
    ner_sets = []

    for st in sentence_sets:
        st_ = []
        pt_ = []
        nt_ = []
        st_len = []
        for s in st:
            s_ = [word2id["_GOO"]]
            for w in s:
                if (w in word2id):
                    s_.append(word2id[w])
                else:
                    s_.append(word2id["_UNK"])
            p_ = []
            tagged_set = nltk.pos_tag(s)
            for ss, t in tagged_set:
                if (t in pos2id):
                    p_.append(pos2id[t])
                else:
                    p_.append(pos2id["_UNK"])
            p_.append(pos2id["_EOS"])
            n_ = []
            ner_set = tree2conlltags(nltk.ne_chunk(tagged_set))
            for i, j, k in ner_set:
                if (j in ner2id):
                    n_.append(ner2id[j])
                else:
                    n_.append(ner2id["_UNK"])
            n_.append(ner2id["_EOS"])

            s_.append(word2id["_EOS"])

            s_ = s_[: max_sent_len]
            if (len(s_) < max_sent_len):
                s_len = len(s_) - 1
                for i in range(max_sent_len - len(s_)): s_.append(word2id["_PAD"])
            else:
                s_[-1] = word2id["_EOS"]
                s_len = max_sent_len - 1

            p_ = p_[: max_sent_len]
            if (len(p_) < max_sent_len):
                for i in range(max_sent_len - len(p_)): p_.append(pos2id["_PAD"])
            n_ = n_[: max_sent_len]
            if (len(n_) < max_sent_len):
                for i in range(max_sent_len - len(n_)): n_.append(ner2id["_PAD"])
            st_.append(s_)
            st_len.append(s_len)
            pt_.append(p_)
            nt_.append(n_)
        pos_sets.append(pt_)
        ner_sets.append(nt_)
        sent_sets.append(st_)
        sent_len_sets.append(st_len)
    return list(zip(sent_sets, pos_sets, ner_sets)), sent_len_sets


def get_vocab(training_set, vocab_size_threshold=5):
    """Get the vocabulary from the training set"""
    vocab = []
    for st in training_set:
        for s in st:
            vocab.extend(s)

    vocab = Counter(vocab)
    vocab_truncate = [w for w in vocab if vocab[w] >= vocab_size_threshold]

    word2id = {"_GOO": 0, "_EOS": 1, "_PAD": 2, "_UNK": 3}
    id2word = {0: "_GOO", 1: "_EOS", 2: "_PAD", 3: "_UNK"}

    i = len(word2id)
    for w in vocab_truncate:
        word2id[w] = i
        id2word[i] = w
        i += 1

    assert (len(word2id) == len(id2word))
    print("vocabulary size: %d" % len(word2id))

    tags = []
    ners = []
    for st in training_set:
        for s in st:
            tagged_st = nltk.pos_tag(s)
            for s, t in tagged_st:
                tags.append(t)
            ner_st = tree2conlltags(nltk.ne_chunk(tagged_st))
            for i, j, k in ner_st:
                ners.append(j)

    tags = Counter(tags)
    ners = Counter(ners)

    pos2id = {"_GOO": 0, "_EOS": 1, "_PAD": 2, "_UNK": 3}
    id2pos = {0: "_GOO", 1: "_EOS", 2: "_PAD", 3: "_UNK"}

    ner2id = {"_GOO": 0, "_EOS": 1, "_PAD": 2, "_UNK": 3}
    id2ner = {0: "_GOO", 1: "_EOS", 2: "_PAD", 3: "_UNK"}

    i = len(pos2id)
    for t in tags:
        pos2id[t] = i
        id2pos[i] = t
        i += 1

    i = len(ner2id)
    for n in ners:
        ner2id[n] = i
        id2ner[i] = n
        i += 1

    assert (len(pos2id) == len(id2pos))
    assert (len(ner2id) == len(id2ner))
    print("pos size: %d" % len(pos2id))
    print("ner size: %d" % len(ner2id))

    return word2id, id2word, pos2id, id2pos, ner2id, id2ner

# class MDataLoader:
#     def __init__(self):
#         self.path = config.dataset_path
#         self.test_index_path = config.dataset_test_index_path
#         self.sentences = []
#         self.tokens = []
#         self.test_index = []
#         self.train_tokens = []
#         self.test_tokens = []
#
#     def load(self):
#         with open(self.path, encoding='utf-8') as fd:
#             lines = fd.readlines()
#         for line in lines:
#             s1, s2 = line[:-1].split('\t')
#             self.sentences.append((s1, s2))
#             self.tokens.append((word_tokenize(s1), word_tokenize(s2)))
#         self.all_num = len(self.sentences)
#
#         with open(self.test_index_path, encoding='utf-8') as fd:
#             lines = fd.readlines()
#         for line in lines:
#             index = int(line[:-1])
#             self.test_index.append(index)
#         self.test_num = len(self.test_index)
#
#         print(len(self.tokens))
#
#     def spilt_train_test(self):
#         flag_table = [True] * self.all_num
#         for index in self.test_index:
#             if index >= self.all_num:
#                 continue
#             flag_table[index] = False
#
#         for index, token in enumerate(self.tokens):
#             if flag_table[index]:
#                 self.train_tokens.append(token)
#             else:
#                 self.test_tokens.append(token)
