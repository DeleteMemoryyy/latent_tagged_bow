import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import LSTMStateTuple
from loss import sequence_loss, _copy_loss

from decoder import decode


def create_cell(name, state_size, drop_out, no_residual=False):
    """Create a LSTM cell"""
    # This one should be the fastest
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(state_size, name=name, dtype=tf.float32,
                                             initializer=tf.compat.v1.random_normal_initializer(stddev=0.05))
    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1. - drop_out)
    if (no_residual == False):
        print('use residual ... ')
        cell = tf.compat.v1.nn.rnn_cell.ResidualWrapper(cell)
    else:
        print('not use residual')
    return cell


def bow_predict_seq_tag(vocab_size,
                        enc_batch_size,
                        enc_outputs,
                        enc_lens,
                        max_len,
                        max_src2tgt_word=3):
    """bow prediction as sequence tagging

    Let each word from the source sentence predict its k nearest neighbors
    """
    bow_topk_prob = tf.zeros([enc_batch_size, vocab_size])
    gumbel_topk_prob = tf.zeros([enc_batch_size, vocab_size])
    seq_neighbor_ind = []
    seq_neighbor_prob = []

    for i in range(max_src2tgt_word):
        bow_trans = tf.compat.v1.layers.Dense(500, name="bow_src2tgt_trans_%d" % i,
                                              kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                              bias_initializer=tf.constant_initializer(0.))
        bow_proj = tf.compat.v1.layers.Dense(vocab_size, name="bow_src2tgt_proj_%d" % i,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                             bias_initializer=tf.constant_initializer(0.))
        bow_logits = bow_proj(bow_trans(enc_outputs))  # [B, T, S] -> [B, T, V]

        # mixture of softmax probability
        bow_prob = tf.nn.softmax(bow_logits)
        pred_mask = tf.expand_dims(
            tf.sequence_mask(enc_lens, max_len, tf.float32), [2])  # [B, T, 1]
        bow_prob *= pred_mask  # [B, T, V]
        bow_topk_prob += tf.reduce_sum(bow_prob, 1)

        # record neighbor prediction
        neighbor_ind = tf.argmax(bow_prob, 2)  # [B, T]
        seq_neighbor_ind.append(neighbor_ind)
        neighbor_prob = tf.reduce_max(bow_prob, 2)  # [B, T]
        seq_neighbor_prob.append(neighbor_prob)

        gumbel_topk_prob += tf.reduce_sum(bow_prob, 1)

    seq_neighbor_ind = tf.stack(seq_neighbor_ind, 2)  # [B, T, N]
    seq_neighbor_prob = tf.stack(seq_neighbor_prob, 2)  # [B, T, N]
    return bow_topk_prob, gumbel_topk_prob, seq_neighbor_ind, seq_neighbor_prob


def _enc_target_list_to_khot(enc_targets, vocab_size, pad_id):
    """Convert a batch of target list to k-hot vectors"""
    enc_targets = tf.one_hot(enc_targets, vocab_size)  # [B, BOW, V]
    enc_targets = tf.reduce_sum(input_tensor=enc_targets, axis=1)  # [B, V]
    enc_target_mask = 1. - tf.one_hot(
        [pad_id], vocab_size, dtype=tf.float32)
    enc_targets *= enc_target_mask
    return enc_targets


def bow_train_monitor(
        bow_topk_prob, pred_ind, vocab_size, batch_size, enc_targets):
    """Precision and recall for the bow model, as well as their supports"""
    pred_one_hot = tf.one_hot(pred_ind, vocab_size)
    pred_one_hot = tf.reduce_sum(input_tensor=pred_one_hot, axis=1)
    pred_one_hot = tf.cast(pred_one_hot, tf.bool)
    pred_topk_support = tf.reduce_sum(input_tensor=tf.cast(pred_one_hot, tf.float32))

    pred_confident = tf.cast(bow_topk_prob > 0.5, tf.bool)  # approximate
    pred_confident_support = tf.reduce_sum(input_tensor=tf.cast(pred_confident, tf.float32))
    predict_average_confident = pred_confident_support / \
                                tf.cast(batch_size, tf.float32)

    enc_targets_ = tf.cast(enc_targets, tf.bool)
    pred_overlap_topk = tf.reduce_sum(
        input_tensor=tf.cast(tf.logical_and(pred_one_hot, enc_targets_), tf.float32))

    pred_overlap_confident = tf.reduce_sum(
        input_tensor=tf.cast(tf.logical_and(pred_confident, enc_targets_), tf.float32))

    target_support = tf.reduce_sum(input_tensor=enc_targets)
    precision_confident = pred_overlap_confident / (pred_confident_support + 1)
    recall_confident = pred_overlap_confident / (target_support + 1)
    precision_topk = pred_overlap_topk / (pred_topk_support + 1)
    recall_topk = pred_overlap_topk / (target_support + 1)
    target_average = target_support / tf.cast(batch_size, tf.float32)

    metric_dict = {"pred_overlap_topk": pred_overlap_topk,
                   "pred_overlap_confident": pred_overlap_confident,

                   "pred_topk_support": pred_topk_support,
                   "pred_confident_support": pred_confident_support,
                   "target_support": target_support,

                   "predict_average_confident": predict_average_confident,
                   "target_average": target_average,

                   # "pred_prob": pred_prob,
                   # "pred_prob_unnorm": pred_prob_unnorm,
                   # "pred_ind": pred_ind,

                   "precision_confident": precision_confident,
                   "recall_confident": recall_confident,
                   "precision_topk": precision_topk,
                   "recall_topk": recall_topk}
    return metric_dict


def enc_loss_fn(enc_targets, bow_topk_prob):
    """Different encoder loss wrapper"""
    # NLL loss
    # if (bow_loss_fn == "nll"):
    enc_loss = - enc_targets * tf.math.log(bow_topk_prob + 1e-6)
    enc_loss_norm = tf.reduce_sum(input_tensor=enc_targets, axis=1) + 1.0
    enc_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=enc_loss, axis=1) / enc_loss_norm)

    # # cross entropy loss
    # # normalize -- This is not so strict, but be it for now
    # elif (bow_loss_fn == "crossent"):
    #     bow_topk_prob /= float(max_enc_bow)
    #     enc_loss = - (enc_targets * tf.math.log(bow_topk_prob + 1e-6) +
    #                   (1 - enc_targets) * tf.math.log(1 - bow_topk_prob + 1e-6))
    #     enc_loss = tf.reduce_mean(
    #         input_tensor=tf.reduce_sum(input_tensor=enc_loss, axis=1) / tf.reduce_sum(input_tensor=enc_targets, axis=1))
    #
    # # L1 distance loss
    # # L1 distance = total variation of the two distributions
    # elif (bow_loss_fn == "l1"):
    #     enc_loss = tf.compat.v1.losses.absolute_difference(enc_targets, bow_topk_prob)
    return enc_loss


################################################################################
## Auxiliary functions

def bow_gumbel_topk_sampling(bow_topk_prob, embedding_matrix, sample_size,
                             vocab_size):
    """Given the soft `bow_topk_prob` k_hot vector, sample `sample_size` locations
  from it, build the soft memory one the fly"""
    # Not differentiable here
    prob, ind = tf.nn.top_k(bow_topk_prob, sample_size)  # [B, sample_size]
    ind_one_hot = tf.one_hot(ind, vocab_size)  # [B, sample_size, V]

    # Differentiable below
    # [B, 1, V]
    bow_topk_prob_ = tf.expand_dims(bow_topk_prob, [1])
    # [B, sample_size, V] -> [B, sample_size]
    sample_prob = tf.reduce_sum(input_tensor=bow_topk_prob_ * ind_one_hot, axis=2)
    # [B, sample_size, S]
    sample_memory = tf.nn.embedding_lookup(params=embedding_matrix, ids=ind)
    sample_memory *= tf.expand_dims(sample_prob, [2])

    return ind, sample_prob, sample_memory


def _calculate_dec_out_mem_ratio(
        dec_outputs, sample_ind, vocab_size, pad_id, start_id, end_id):
    """Calculate what portion of the output is in the memory"""
    # dec_outputs.shape = [B, T]
    dec_outputs_bow = tf.one_hot(dec_outputs, vocab_size, dtype=tf.float32)
    dec_outputs_bow = tf.reduce_sum(input_tensor=dec_outputs_bow, axis=1)  # [B, V]
    mask = tf.one_hot([start_id, end_id, pad_id], vocab_size, dtype=tf.float32)
    mask = 1. - tf.reduce_sum(input_tensor=mask, axis=0)  # [V]
    dec_outputs_bow *= tf.expand_dims(mask, [0])

    sample_ind = tf.one_hot(sample_ind, vocab_size, dtype=tf.float32)  # [B, M, V]
    sample_ind = tf.reduce_sum(input_tensor=sample_ind, axis=1)  # [B, V]

    overlap = tf.reduce_sum(input_tensor=dec_outputs_bow * sample_ind, axis=1)  # [B]
    dec_output_support = tf.reduce_sum(input_tensor=dec_outputs_bow, axis=1)  # [B]
    ratio = overlap / dec_output_support

    dec_out_mem_ratio = {
        "words_from_mem": tf.reduce_mean(input_tensor=overlap),
        "dec_output_bow_cnt": tf.reduce_mean(input_tensor=dec_output_support),
        "dec_mem_ratio": tf.reduce_mean(input_tensor=ratio)}
    return dec_out_mem_ratio


################################################################################
## Model class

class LatentBow(object):
    """The latent bow model

  The encoder will encode the souce into b and z:
    b = bow model, regularized by the bow loss
    z = content model

  Then we sample from b with gumbel topk, and construct a dynamic memory on the
  fly with the sample. The decoder will be conditioned on this memory
  """

    def __init__(self, config):
        """Initialization"""
        self.mode = config.model_mode
        self.model_name = config.model_name
        self.vocab_size = config.vocab_size
        self.max_enc_bow = config.max_enc_bow
        self.sample_size = config.sample_size
        self.state_size = config.state_size
        self.enc_layers = config.enc_layers
        self.learning_rate = config.learning_rate
        self.learning_rate_enc = config.learning_rate_enc
        self.learning_rate_dec = config.learning_rate_dec
        self.drop_out_config = config.drop_out
        self.optimizer = config.optimizer
        self.dec_start_id = config.dec_start_id
        self.dec_end_id = config.dec_end_id
        self.pad_id = config.pad_id
        self.stop_words = config.stop_words
        self.lambda_enc_loss = config.lambda_enc_loss
        self.no_residual = config.no_residual
        self.copy = config.copy
        self.bow_cond = config.bow_cond
        self.bow_cond_gate = config.bow_cond_gate
        self.num_pointers = config.num_pointers
        return

    def build(self):
        """Build the model"""
        print("Building the Latent BOW - sequence to sequence model ... ")

        vocab_size = self.vocab_size
        state_size = self.state_size
        enc_layers = self.enc_layers
        max_enc_bow = self.max_enc_bow
        lambda_enc_loss = self.lambda_enc_loss

        # Placeholders
        with tf.compat.v1.name_scope("placeholders"):
            enc_inputs = tf.compat.v1.placeholder(tf.int32, [None, None], "enc_inputs")
            enc_lens = tf.compat.v1.placeholder(tf.int32, [None], "enc_lens")
            self.drop_out = tf.compat.v1.placeholder(tf.float32, (), "drop_out")

            self.enc_inputs = enc_inputs
            self.enc_lens = enc_lens

            enc_targets = tf.compat.v1.placeholder(tf.int32, [None, None], "enc_targets")
            dec_inputs = tf.compat.v1.placeholder(tf.int32, [None, None], "dec_inputs")
            dec_targets = tf.compat.v1.placeholder(tf.int32, [None, None], "dec_targets")
            dec_lens = tf.compat.v1.placeholder(tf.int32, [None], "dec_lens")

            self.enc_targets = enc_targets
            self.dec_inputs = dec_inputs
            self.dec_targets = dec_targets
            self.dec_lens = dec_lens

        batch_size = tf.shape(input=enc_inputs)[0]
        max_len = tf.shape(input=enc_inputs)[1]

        # Embedding
        with tf.compat.v1.variable_scope("embeddings"):
            embedding_matrix = tf.compat.v1.get_variable(
                name="embedding_matrix",
                shape=[vocab_size, state_size],
                dtype=tf.float32,
                initializer=tf.compat.v1.random_normal_initializer(stddev=0.05))

            enc_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=enc_inputs)
            dec_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=dec_inputs)

        # Encoder
        with tf.compat.v1.variable_scope("encoder"):
            # TODO: residual LSTM, layer normalization
            enc_cell = [create_cell(
                "enc-%d" % i, state_size, self.drop_out, self.no_residual)
                for i in range(enc_layers)]
            enc_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(enc_cell)
            enc_outputs, enc_state = tf.compat.v1.nn.dynamic_rnn(enc_cell, enc_inputs,
                                                                 sequence_length=enc_lens, dtype=tf.float32)

        # Encoder bow prediction
        with tf.compat.v1.variable_scope("bow_output"):
            bow_topk_prob, gumbel_topk_prob, seq_neighbor_ind, seq_neighbor_prob = \
                bow_predict_seq_tag(vocab_size, batch_size, enc_outputs, enc_lens,
                                    max_len)
            seq_neighbor_output = {"seq_neighbor_ind": seq_neighbor_ind,
                                   "seq_neighbor_prob": seq_neighbor_prob}

        # Encoder output, loss and metrics
        with tf.compat.v1.name_scope("enc_output"):
            # top k prediction
            bow_pred_prob, pred_ind = tf.nn.top_k(bow_topk_prob, max_enc_bow)

            # loss function
            enc_targets = _enc_target_list_to_khot(
                enc_targets, vocab_size, self.pad_id)
            enc_loss = enc_loss_fn(
                enc_targets, bow_topk_prob)
            self.train_output = {"enc_loss": enc_loss}

            # performance monitor
            bow_metrics_dict = bow_train_monitor(
                bow_topk_prob, pred_ind, vocab_size, batch_size, enc_targets)
            self.train_output.update(bow_metrics_dict)

        # Encoder soft sampling
        with tf.compat.v1.name_scope("gumbel_topk_sampling"):
            sample_ind, sample_prob, sample_memory = bow_gumbel_topk_sampling(gumbel_topk_prob, embedding_matrix,
                                                                              self.sample_size, vocab_size)

            sample_memory_lens = tf.ones(batch_size, tf.int32) * self.sample_size
            sample_memory_avg = tf.reduce_mean(input_tensor=sample_memory, axis=1)  # [B, S]

            sample_memory_output = {"bow_pred_ind": pred_ind,
                                    "bow_pred_prob": bow_pred_prob,
                                    "sample_memory_ind": sample_ind,
                                    "sample_memory_prob": sample_prob}

        # Decoder
        # The initial state of the decoder =
        #   encoder meaning vector z + encoder bow vector b
        with tf.compat.v1.variable_scope("decoder"):
            dec_cell = [create_cell(
                "dec-%d" % i, state_size, self.drop_out, self.no_residual)
                for i in range(enc_layers)]
            dec_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(dec_cell)
            dec_proj = tf.compat.v1.layers.Dense(vocab_size, name="dec_proj",
                                                 kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.05),
                                                 bias_initializer=tf.compat.v1.constant_initializer(0.))
            dec_ptr_k_proj = [
                tf.compat.v1.layers.Dense(state_size, name="dec_ptr_k_proj_%d" % pi,
                                          kernel_initializer=tf.compat.v1.random_normal_initializer(stddev=0.05),
                                          bias_initializer=tf.compat.v1.constant_initializer(0.))
                for pi in range(self.num_pointers)]
            dec_ptr_g_proj = tf.compat.v1.layers.Dense(1, name="dec_ptr_g_proj",
                                                       kernel_initializer=tf.compat.v1.random_normal_initializer(
                                                           stddev=0.05),
                                                       bias_initializer=tf.compat.v1.constant_initializer(0.),
                                                       activation=tf.nn.sigmoid)
            bow_cond_gate_proj = tf.compat.v1.layers.Dense(1, name="bow_cond_gate_proj",
                                                           kernel_initializer=tf.compat.v1.random_normal_initializer(
                                                               stddev=0.05),
                                                           bias_initializer=tf.compat.v1.constant_initializer(0.),
                                                           activation=tf.nn.sigmoid)

            dec_init_state = []
            for l in range(enc_layers):
                dec_init_state.append(LSTMStateTuple(c=enc_state[0].c,
                                                     h=enc_state[0].h + sample_memory_avg))
            dec_init_state = tuple(dec_init_state)

            # if(enc_layers == 2):
            #   dec_init_state = (LSTMStateTuple( c=enc_state[0].c,
            #                                     h=enc_state[0].h + sample_memory_avg),
            #                     LSTMStateTuple( c=enc_state[1].c,
            #                                     h=enc_state[1].h + sample_memory_avg) )
            # elif(enc_layers == 4):
            #   dec_init_state = (LSTMStateTuple(c=enc_state[0].c,
            #                       h=enc_state[0].h + sample_memory_avg),
            #                     LSTMStateTuple( c=enc_state[1].c,
            #                       h=enc_state[1].h + sample_memory_avg) )
            # else: raise Exception('enc_layers not in [2, 4]')

            # source_attn
            # [B, M + T, S]
            dec_memory = [sample_memory, enc_outputs]
            dec_mem_len = [sample_memory_lens, enc_lens]
            dec_max_mem_len = [self.sample_size, max_len]

            if (self.bow_cond):
                bow_cond = sample_memory_avg
            else:
                bow_cond = None

            if (self.bow_cond_gate == False): bow_cond_gate_proj = None

            (dec_outputs_predict, dec_logits_train, dec_prob_train, pointer_ent,
             avg_max_ptr, avg_num_copy) = decode(
                self.dec_start_id, dec_inputs,
                dec_cell, dec_proj, embedding_matrix,
                dec_init_state, dec_memory, dec_mem_len, dec_max_mem_len,
                batch_size, max_len, state_size, multi_source=True, copy=self.copy, copy_ind=sample_ind,
                dec_ptr_g_proj=dec_ptr_g_proj, dec_ptr_k_proj=dec_ptr_k_proj,
                bow_cond=bow_cond, bow_cond_gate_proj=bow_cond_gate_proj)

        # model saver, before the optimizer
        # all_variables = slim.get_variables_to_restore()
        # model_variables = [var for var in all_variables
        #                    if var.name.split("/")[0] == self.model_name]
        # print("%s model, variable list:" % self.model_name)
        # for v in model_variables: print("  %s" % v.name)
        # self.model_saver = tf.compat.v1.train.Saver(model_variables, max_to_keep=3)

        with tf.compat.v1.variable_scope("optimizer"):
            optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        # decoder output, training and inference, combined with encoder loss
        with tf.compat.v1.name_scope("dec_output"):
            dec_mask = tf.sequence_mask(dec_lens, max_len, dtype=tf.float32)
            if (self.copy == False):
                dec_loss = sequence_loss(
                    dec_logits_train, dec_targets, dec_mask)
            else:
                dec_loss = _copy_loss(dec_prob_train, dec_targets, dec_mask)

            loss = dec_loss + lambda_enc_loss * enc_loss
            train_op = optimizer.minimize(loss)

            dec_output = {"train_op": train_op, "dec_loss": dec_loss, "loss": loss}
            self.train_output.update(dec_output)
            if (self.copy):
                pointer_ent = \
                    tf.reduce_sum(input_tensor=pointer_ent * dec_mask) / tf.reduce_sum(input_tensor=dec_mask)
                self.train_output['pointer_ent'] = pointer_ent
                avg_max_ptr = \
                    tf.reduce_sum(input_tensor=avg_max_ptr * dec_mask) / tf.reduce_sum(input_tensor=dec_mask)
                self.train_output['avg_max_ptr'] = avg_max_ptr
                avg_num_copy = tf.reduce_sum(input_tensor=avg_num_copy * dec_mask, axis=1)
                avg_num_copy = tf.reduce_mean(input_tensor=avg_num_copy)
                self.train_output['avg_num_copy'] = avg_num_copy

            self.infer_output = {"dec_predict": dec_outputs_predict}
            dec_out_mem_ratio = _calculate_dec_out_mem_ratio(dec_outputs_predict,
                                                             sample_ind, vocab_size, self.pad_id, self.dec_start_id,
                                                             self.dec_end_id)
            self.infer_output.update(dec_out_mem_ratio)
            self.infer_output.update(sample_memory_output)
            self.infer_output.update(seq_neighbor_output)
        return

    def train_step(self, sess, batch_dict, ei):
        """Single step training"""
        feed_dict = {self.enc_inputs: batch_dict["enc_inputs"],
                     self.enc_lens: batch_dict["enc_lens"],
                     self.enc_targets: batch_dict["enc_targets"],
                     self.dec_inputs: batch_dict["dec_inputs"],
                     self.dec_targets: batch_dict["dec_targets"],
                     self.dec_lens: batch_dict["dec_lens"],
                     self.drop_out: self.drop_out_config}
        output_dict = sess.run(self.train_output, feed_dict=feed_dict)
        return output_dict

    def predict(self, sess, batch_dict):
        """Single step prediction"""
        feed_dict = {self.enc_inputs: batch_dict["enc_inputs"],
                     self.enc_lens: batch_dict["enc_lens"],
                     self.drop_out: 0.}
        output_dict = sess.run(self.infer_output, feed_dict=feed_dict)
        return output_dict
