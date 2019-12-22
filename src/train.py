import os
import time
import tensorflow as tf
import numpy as np

from nltk.translate.bleu_score import corpus_bleu


def _cut_eos(predict_batch, eos_id):
    """cut the eos in predict sentences"""
    pred = []
    for s in predict_batch:
        s_ = []
        for w in s:
            if (w == eos_id): break
            s_.append(w)
        pred.append(s_)
    return pred


class TrainingLog(object):
    def __init__(self, config):
        # self.model = config.model_name
        self.output_path = config.output_path

        self.log = {
            "loss": [],
            "enc_loss": [],
            "dec_loss": [],

            "pred_overlap_topk": [],
            "pred_overlap_confident": [],

            "pred_topk_support": [],
            "pred_confident_support": [],
            "target_support": [],

            "predict_average_confident": [],
            "target_average": [],
            'pointer_ent': [],
            'avg_max_ptr': [],
            'avg_num_copy': [],

            "precision_confident": [],
            "recall_confident": [0],
            "precision_topk": [],
            "recall_topk": []}

    def update(self, output_dict):
        """Update the log"""
        for l in self.log:
            if (l in output_dict): self.log[l].append(output_dict[l])
        return

    def print(self):
        """Print out the log"""
        s = ""
        for l in self.log: s += "%s: %.4f, " % (l, np.average(self.log[l]))
        print(s)
        print("")
        return

    def write(self, ei, log_metrics=None):
        """Write the log for current epoch"""
        log_path = self.output_path + "epoch_%d.log" % ei
        print("Writing epoch log to %s ... " % log_path)
        with open(log_path, "w", encoding='utf-8') as fd:
            log_len = len(self.log[list(self.log.keys())[0]])
            for i in range(log_len):
                for m in self.log:
                    if (log_metrics == None):
                        fd.write("%s: %.4f\t" % (m, self.log[m][i]))
                    else:
                        if (m in log_metrics): fd.write("%s: %.4f\t" % (m, self.log[m][i]))
                fd.write("\n")
        return

    def reset(self):
        """Reset the log"""
        for l in self.log:
            self.log[l] = []
        return


class Controller(object):
    """The training, validation and evaluation controller"""

    def __init__(self, config):
        """Initialization from the configuration"""
        self.model_name = 'latent_bow'
        self.model_name_version = config.model_name + "_" + config.model_version
        self.start_epoch = config.start_epoch
        self.num_epoch = config.num_epoch
        self.write_output = config.write_output
        self.batch_size = config.batch_size
        self.print_interval = config.train_print_interval
        self.gpu_id = config.gpu_id
        self.drop_out = config.drop_out
        self.dec_start_id = config.dec_start_id
        self.dec_end_id = config.dec_end_id
        self.model_path = config.model_path
        self.output_path = config.output_path
        self.random_seed = config.random_seed
        self.train_log = TrainingLog(config)
        self.id2word = None
        self.target_metrics = config.target_metrics
        self.eval_metrics_list = config.eval_metrics_list
        self.log_metrics = config.log_metrics
        return

    def train(self, model, dset):
        """Train the model with the controller"""
        print("Start training ...")

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        tf.compat.v1.set_random_seed(self.random_seed)

        self.id2word = dset.id2word

        start_epoch = self.start_epoch
        num_epoch = self.num_epoch
        batch_size = self.batch_size
        model_name = self.model_name
        drop_out = self.drop_out
        print_interval = self.print_interval
        train_log = self.train_log
        target_metrics = self.target_metrics
        model_name_version = self.model_name_version

        gpu_config = tf.compat.v1.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=gpu_config)
        sess.run(tf.compat.v1.global_variables_initializer())

        # training preparation
        num_batches = dset.num_batches(batch_size, "train")
        best_target_metrics = -100000
        best_epoch = -1
        print("%d batches in total" % num_batches)
        print("metrics of first 200 batchs are not reliable ")

        for ei in range(start_epoch, start_epoch + num_epoch):
            start_time = time.time()

            for bi in range(num_batches):
                # for bi in range(100):
                batch_dict = dset.next_batch("train", batch_size, model_name)
                batch_dict["drop_out"] = drop_out
                output_dict = model.train_step(sess, batch_dict, ei)
                train_log.update(output_dict)

                if (bi % 20 == 0):
                    print(".", end=" ", flush=True)
                if (bi % print_interval == 0):
                    print("\n%s: e-%d, b-%d, t-%.2f" %
                          (model_name_version, ei, bi, time.time() - start_time))
                    train_log.print()

            print("\n\nepoch %d training finished" % ei)

            if (ei >= 0):
                metrics_dict = self.eval(model, dset, sess, "test", ei=ei)
                if (metrics_dict[target_metrics] > best_target_metrics):
                    best_epoch = ei
                    print("increase validation %s from %.4f to %.4f, update model" %
                          (target_metrics, best_target_metrics, metrics_dict[target_metrics]))
                    # TODO: Save model
                    # save_path = self.model_path + "/model-e%d.ckpt" % ei
                    # if (self.save_ckpt):
                    #     model.model_saver.save(sess, save_path)
                    #     print("saving model to %s" % save_path)
                    best_target_metrics = metrics_dict[target_metrics]
                else:
                    print("no performance increase, keep the best model at epoch %d" %
                          best_epoch)
                    print("best %s: %.4f" % (target_metrics, best_target_metrics))

            print("\nepoch %d, time cost %.2f\n" % (ei, time.time() - start_time))
            train_log.print()
            if (self.write_output): train_log.write(ei, self.log_metrics)
            train_log.reset()
            print("")
        return

    def eval_metrics(self, sess, output_dict, batch_dict):
        """"""
        metrics_dict = {}
        if ("bleu" in self.eval_metrics_list):
            metrics_dict.update(
                {"bleu_1": -1, "bleu_2": -1, "bleu_3": -1, "bleu_4": -1})

        # generated BLEU v.s. reference, a measure of quality
        predicts = _cut_eos(output_dict["dec_predict"], self.dec_end_id)
        reference = batch_dict["references"]

        if ("bleu" in self.eval_metrics_list):
            metrics_dict["bleu_1"] = corpus_bleu(
                reference, predicts, weights=(1, 0, 0, 0))
            metrics_dict["bleu_2"] = corpus_bleu(
                reference, predicts, weights=(0.5, 0.5, 0, 0))
            metrics_dict["bleu_3"] = corpus_bleu(
                reference, predicts, weights=(0.33, 0.33, 0.34, 0))
            metrics_dict["bleu_4"] = corpus_bleu(
                reference, predicts, weights=(0.25, 0.25, 0.25, 0.25))

        # matching score given by a matching model, semantic similarity

        return metrics_dict

    def encoder_eval(self, model, dset, sess, mode):
        """Only evaluate the encoder for the bow_seq2seq_2seq model"""
        print("Evaluating the encoder ... ")

        start_time = time.time()
        batch_size = self.batch_size
        model_name = self.model_name

        num_batches = dset.num_batches(batch_size, mode)
        print("%d batches in total" % num_batches)

        metrics_dict = {"enc_infer_overlap": [],
                        "enc_infer_pred_support": [],
                        "enc_infer_target_support": [],
                        "enc_infer_precision": [],
                        "enc_infer_recall": []}

        for bi in range(num_batches):
            batch_dict = dset.next_batch(mode, batch_size, model_name)
            output_dict = model.enc_infer_step(sess, batch_dict)

            for m in output_dict:
                if (m in metrics_dict): metrics_dict[m].append(output_dict[m])

        dset.print_predict_seq2paraphrase(output_dict, batch_dict)

        for m in metrics_dict:
            metrics_dict[m] = np.average(metrics_dict[m])
            print("%s: %.4f" % (m, metrics_dict[m]))
        print("time cost: %.2fs" % (time.time() - start_time))
        print("")
        return metrics_dict

    def eval_generate(self, model, dset, sess, mode, ei=-1):
        """Validation or evaluation"""
        print("Evaluating ... ")

        start_time = time.time()
        batch_size = self.batch_size
        model_name = self.model_name

        num_batches = dset.num_batches(batch_size, mode)
        print("%d batches in total" % num_batches)

        metrics_dict = {}
        if ("bleu" in self.eval_metrics_list):
            metrics_dict.update(
                {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []})

        if (self.write_output):
            output_fd = open(self.output_path + "output_e%d.txt" % ei, "w", encoding='utf-8')
        # pred_batch = np.random.randint(0, num_batches)
        pred_batch = 0

        references = []
        dec_outputs = []

        for bi in range(num_batches):
            # for bi in range(50):
            batch_dict = dset.next_batch(mode, batch_size, model_name)
            references.extend(batch_dict['references'])
            output_dict = model.predict(sess, batch_dict)
            dec_outputs.extend(
                _cut_eos(output_dict["dec_predict"], self.dec_end_id))

            dset.print_predict(output_dict, batch_dict,
                               output_fd)

        if (bi == pred_batch):
            print("")
            dset.print_predict(output_dict, batch_dict, None)

        metrics_dict_update = self.eval_metrics(sess, output_dict, batch_dict)
        metrics_dict_update.update(output_dict)
        for m in metrics_dict_update:
            if (m in metrics_dict):
                if (m[:4] != "dist"):
                    metrics_dict[m].append(metrics_dict_update[m])
        if (bi % 20 == 0): print(".", end=" ", flush=True)

        print("")
        if (self.write_output): output_fd.close()

        # print('corpus level bleu:')
        # print(references[0])
        # print(dec_outputs[0])
        # print('bleu 1: %.4f' %
        #   corpus_bleu(references, dec_outputs, weights=[1, 0, 0, 0]))
        # print('bleu 2: %.4f' %
        #   corpus_bleu(references, dec_outputs, weights=[1, 1, 0, 0]))
        # print('bleu 3: %.4f' %
        #   corpus_bleu(references, dec_outputs, weights=[1, 1, 1, 0]))
        # print('bleu 4: %.4f' %
        #   corpus_bleu(references, dec_outputs, weights=[1, 1, 1, 1]))

        print('sentence level:')
        for m in metrics_dict:
            if (m[:4] != "dist"):
                metrics_dict[m] = np.average(metrics_dict[m])
            print("%s: %.4f" % (m, metrics_dict[m]))

        print("time cost: %.2fs" % (time.time() - start_time))
        print("")
        return metrics_dict

    def eval(self, model, dset, sess, mode, ei=-1):
        metrics_dict = self.eval_generate(model, dset, sess, mode, ei)
        return metrics_dict
