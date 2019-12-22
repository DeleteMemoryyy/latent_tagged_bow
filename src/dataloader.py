import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from config import config


class MDataLoader:
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


def build_batch_bow_seq2seq_eval(
        data_batch, len_batch, stop_words, max_enc_bow, max_dec_bow, pad_id,
        single_ref=False):
    """Build evaluation batch, basically the same as the seq2seq setting"""
    enc_inputs = []
    enc_lens = []
    references = []
    dec_golden_bow = []
    dec_bow = []
    dec_bow_len = []

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

    for st, slen in zip(data_batch, len_batch):
        inp = st[0][: -1]
        len_inp = slen[0]
        if (single_ref):
            ref = [s_[1: s_len_] for s_, s_len_ in zip(st[-1:], slen[-1:])]
        else:
            ref = [s_[1: s_len_] for s_, s_len_ in zip(st[1:], slen[1:])]

        golden_bow = []
        for r in ref: golden_bow.extend(r)
        golden_bow = set(golden_bow) - stop_words
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
                  "references": references}
    return batch_dict


def build_batch_bow_seq2seq(data_batch,
                            len_batch,
                            stop_words,
                            max_enc_bow,
                            max_dec_bow,
                            pad_id):
    """Build a training batch for the bow seq2seq model"""
    enc_inputs = []
    enc_lens = []
    enc_targets = []
    dec_bow = []
    dec_bow_len = []
    dec_inputs = []
    dec_targets = []
    dec_lens = []

    def _pad(s_set, max_len, pad_id):
        s_set = list(s_set)[: max_len]
        for i in range(max_len - len(s_set)): s_set.append(pad_id)
        return s_set

    for st, slen in zip(data_batch, len_batch):
        para_bow = set()
        for s in st: para_bow |= set(s)
        para_bow -= stop_words
        para_bow = _pad(para_bow, max_enc_bow, pad_id)

        num_para = len(st)

        for i in range(num_para):
            j = (i + 1) % num_para
            inp = st[i][: -1]
            d_in = st[j][: -1]
            d_out = st[j][1:]
            len_inp = slen[i]
            len_out = slen[j]

            enc_inputs.append(inp)
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
            e_bow = _pad(e_bow, max_enc_bow, pad_id)

            # method 2: repeat bow
            # e_bow = list(e_bow)
            # e_bow_ = []
            # i = 0
            # while(len(e_bow_) < max_enc_bow):
            #   e_bow_.append(e_bow[i])
            #   i = (i + 1) % len(e_bow)
            # e_bow = e_bow_

            enc_targets.append(e_bow)

            # enc_targets.append(para_bow)

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
                  "dec_lens": np.array(dec_lens)}
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
        train_sentences = quora_read(self.dataset_path["train"])

        # corpus_statistics(train_sentences)

        train_sentences, dev_sentences = train_dev_split(
            self.dataset, train_sentences, self.train_index_path, self.dev_index_path)

        word2id, id2word = get_vocab(train_sentences)

        train_sentences, train_lens = normalize(
            train_sentences, word2id, self.max_sent_len)
        dev_sentences, dev_lens = normalize(
            dev_sentences, word2id, self.max_sent_len)

        # test_sentences = mscoco_read_json(self.dataset_path["test"])
        # test_sentences, test_lens = normalize(
        #   test_sentences, word2id, self.max_sent_len)

        self.word2id = word2id
        self.id2word = id2word
        self.start_id = word2id["_GOO"]
        self.eos_id = word2id["_EOS"]
        self.unk_id = word2id["_UNK"]
        self.pad_id = word2id["_PAD"]
        self.stop_words = set(
            [word2id[w] if (w in word2id) else self.pad_id for w in self.stop_words])
        self.stop_words |= set(
            [self.start_id, self.eos_id, self.unk_id, self.pad_id])

        self._dataset["train"] = train_sentences
        self._dataset["test"] = dev_sentences
        # self._dataset["test"] = test_sentences
        self._sent_lens["train"] = train_lens
        self._sent_lens["test"] = dev_lens
        # self._sent_lens["test"] = test_lens
        return

    def next_batch(self, setname, batch_size, model_name):
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

    def print_predict(self,
                      output_dict, batch_dict, fd=None):
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

    def print_gumbel(self, output_dict, batch_dict, fd=None):
        """Print the gumbel sampes """
        if (fd == None):
            print_range = 5
        else:
            print_range = len(output_dict[0]["dec_predict"])
        for i in range(print_range):
            dec_out_i = []
            for j in range(len(output_dict)):
                dec_out_i.append(output_dict[j]["dec_predict"][i])
            dec_out_i = np.array(dec_out_i)
            if (np.all((dec_out_i - dec_out_i[0]) == 0)): continue

            out = "inputs:\n"
            out += "    " + self.decode_sent(batch_dict["enc_inputs"][i]) + "\n\n"
            for j in range(len(output_dict)):
                # out += "sample %d, input neighbors:\n" % j
                # out += self.decode_neighbor(batch_dict["enc_inputs"][i],
                #                             output_dict[j]["seq_neighbor_ind"][i],
                #                             output_dict[j]["seq_neighbor_prob"][i])
                # out += "sample %d, enc_output_bow:\n" % j
                # out += "    " + self.decode_sent(output_dict[j]["bow_pred_ind"][i]) + "\n"
                out += "sample %d, enc_sampled_memory:\n" % j
                out += "    " + self.decode_sent(
                    output_dict[j]["sample_memory_ind"][i],
                    prob=output_dict[j]["sample_memory_prob"][i]) + "\n"
                out += "sample %d, dec_outputs:\n" % j
                out += "    " + self.decode_sent(
                    output_dict[j]["dec_predict"][i]) + "\n\n"
            # out += "references:\n"
            # for r in batch_dict["references"][i]:
            #   out += "    " + self.decode_sent(r) + "\n"
            out += "----\n"
            if (fd is not None):
                fd.write(out + "\n")
            else:
                print(out)
        return

    def print_predict_bow_seq2seq(self, output_dict, batch_dict):
        """Print the predicted sentences for the bag of words - sequence to
        sequence model

        Things to print:
          * The input sentence
          * The predicted bow
          * The sample from the bow
          * The predicted sentence
          * The references
        """
        enc_sentences = batch_dict["enc_inputs"]
        enc_outputs = output_dict["pred_ind"]
        dec_samples = output_dict["dec_sample_bow"]
        dec_outputs = output_dict["dec_predict"]
        references = batch_dict["references"]

        for i, (es, eo, ds, do, rf) in enumerate(zip(enc_sentences, enc_outputs,
                                                     dec_samples, dec_outputs, references)):
            print("inputs:")
            print("    " + self.decode_sent(es))
            print("enc_outputs:")
            print("    " + self.decode_sent(eo))
            print("dec_samples:")
            print("    " + self.decode_sent(ds))
            print("dec_outputs:")
            print("    " + self.decode_sent(do))
            print("reference:")
            for r in rf:
                print("    " + self.decode_sent(r))
            print("")

            if (i == 5): break
        return

    def print_predict_seq2seq(
            self, output_dict, batch_dict, fd=None, num_cases=6):
        """Print the predicted sentences for the sequence to sequence model"""
        predict = output_dict["dec_predict"]
        inputs = batch_dict["enc_inputs"]
        references = batch_dict["references"]
        batch_size = output_dict["dec_predict"].shape[0]
        for i in range(batch_size):
            str_out = 'inputs:\n'
            str_out += self.decode_sent(inputs[i]) + '\n'
            str_out += 'outputs:\n'
            str_out += self.decode_sent(predict[i]) + '\n'
            str_out += "references:\n"
            for r in references[i]:
                str_out += self.decode_sent(r) + '\n'
            str_out += '----\n'
            if (i < num_cases): print(str_out)
            if (fd is not None): fd.write(str_out)
        return

    def print_predict_seq2paraphrase(self, output_dict, batch_dict, num_cases=3):
        """Print the predicted sentences, sequence to k sequence model (given a
          sentence, predict all k possible paraphrases)"""
        inputs = batch_dict["enc_inputs"][:3]
        references = batch_dict["references"][:3]
        for i in range(num_cases):
            print("inputs:")
            print("    " + self.decode_sent(inputs[i]))
            pred_para = output_dict["enc_infer_pred"][i]
            print("paraphrase outputs:")
            for p in pred_para:
                print("    " + self.decode_sent(p))
            print("references:")
            for r in references[i]:
                print("    " + self.decode_sent(r))
            print("")
        return

    def print_random_walk(self, random_walk_outputs, batch_dict, num_cases=3):
        """Print the random walk outputs"""
        inputs = batch_dict["enc_inputs"][:3]
        references = batch_dict["references"][:3]
        for i in range(num_cases):
            print("inputs:")
            print("    " + self.decode_sent(inputs[i]))
            for d in random_walk_outputs:
                print("->")
                print("    " + self.decode_sent(d["predict"][i]))
            print("references:")
            for r in references[i]:
                print("    " + self.decode_sent(r))
            print("")
        return

    def print_bow(self, output_dict, batch_dict):
        """Print the bow prediction: the input sentence, the target bow, and the
        predicted bow. """
        enc_sentences = batch_dict["enc_inputs"]
        enc_targets = batch_dict["enc_targets"]
        enc_outputs = output_dict["pred_ind"]

        def _decode_set(s, shared):
            output = []
            for si in s:
                if (si in shared):
                    output.append("[" + self.id2word[si] + "]")
                else:
                    output.append(self.id2word[si])
            return

        for i, (s, t, o) in enumerate(zip(enc_sentences, enc_targets, enc_outputs)):
            if (i in [0, 1, 5, 6, 10, 11, 15, 16]):
                print("inputs:")
                print("    " + self.decode_sent(s))
                shared = set(t) & set(o)
                print("targets:")
                print("    " + _decode_set(set(t) - set([self.pad_id]), shared))
                print("outputs:")
                print("    " + _decode_set(set(o), shared))
                print("")
            if (i == 16): break
        return


# nlp tools
def normalize(sentence_sets, word2id, max_sent_len):
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

    for st in sentence_sets:
        st_ = []
        st_len = []
        for s in st:
            s_ = [word2id["_GOO"]]
            for w in s:
                if (w in word2id):
                    s_.append(word2id[w])
                else:
                    s_.append(word2id["_UNK"])
            s_.append(word2id["_EOS"])

            s_ = s_[: max_sent_len]
            if (len(s_) < max_sent_len):
                s_len = len(s_) - 1
                for i in range(max_sent_len - len(s_)): s_.append(word2id["_PAD"])
            else:
                s_[-1] = word2id["_EOS"]
                s_len = max_sent_len - 1

            st_.append(s_)
            st_len.append(s_len)

        sent_sets.append(st_)
        sent_len_sets.append(st_len)
    return sent_sets, sent_len_sets


def corpus_statistics(sentence_sets, vocab_size_threshold=5):
    """Calculate basic corpus statistics"""
    print("Calculating basic corpus statistics .. ")

    stop_words = set(stopwords.words('english'))

    # size of paraphrase sets
    paraphrase_size = [len(st) for st in sentence_sets]
    paraphrase_size = Counter(paraphrase_size)
    print("paraphrase size, %d different types:" % (len(paraphrase_size)))
    print(paraphrase_size.most_common(10))

    # sentence length
    sentence_lens = []
    sentence_bow_len = []
    paraphrase_bow_len = []
    for st in sentence_sets:
        sentence_lens.extend([len(s) for s in st])
        st_bow = set()
        for s in st:
            s_ = set(s) - stop_words
            sentence_bow_len.append(len(s_))
            st_bow |= s_
        paraphrase_bow_len.append(len(st_bow))

    sent_len_percentile = np.percentile(sentence_lens, [80, 90, 95])
    print("sentence length percentile:")
    print(sent_len_percentile)

    sentence_bow_percentile = np.percentile(sentence_bow_len, [80, 90, 95])
    print("sentence bow length percentile")
    print(sentence_bow_percentile)

    paraphrase_bow_percentile = np.percentile(paraphrase_bow_len, [80, 90, 95])
    print("paraphrase bow length percentile")
    print(paraphrase_bow_percentile)

    # vocabulary
    vocab = []
    for st in sentence_sets:
        for s in st:
            vocab.extend(s)
    vocab = Counter(vocab)
    print("vocabulary size: %d" % len(vocab))
    vocab_truncate = [w for w in vocab if vocab[w] >= vocab_size_threshold]
    print("vocabulary size, occurance >= 5: %d" % len(vocab_truncate))
    return


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
    return word2id, id2word
