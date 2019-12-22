import tensorflow as tf


def beamsearch_decoding():
    """The beam search decoding algorithm, used for inference"""
    return


def multi_source_attention(query, memory, mem_lens, max_mem_len, state_size):
    """Attention to multiple sources

    query: a query vector [B, S]
    memory: a list of memory vectors [[B, M, S], [B, M, S], ... ]
    mem_lens: a list of memory length
    max_mem_len: a list of maximum memory length
    """
    with tf.compat.v1.variable_scope("multi_source_attention"):
        num_memory = len(memory)
        context_vector = []

        for i in range(num_memory):
            query = tf.compat.v1.layers.dense(query, state_size, name="query_proj",
                                              kernel_initializer=tf.compat.v1.random_normal_initializer(
                                                  stddev=0.05),
                                              bias_initializer=tf.compat.v1.constant_initializer(0.),
                                              reuse=tf.compat.v1.AUTO_REUSE)
            context, _ = attention(
                query, memory[i], mem_lens[i], max_mem_len[i])
            context_vector.append(context)

        context_vector = tf.concat(context_vector, axis=1)
        # [B, num_mem * S] -> [B, S]
        context_vector = tf.compat.v1.layers.dense(context_vector, state_size,
                                                   name="mem_output_proj",
                                                   kernel_initializer=tf.compat.v1.random_normal_initializer(
                                                       stddev=0.05),
                                                   bias_initializer=tf.compat.v1.constant_initializer(0.),
                                                   reuse=tf.compat.v1.AUTO_REUSE)

    return (context_vector, None)


def attention(query, memory, mem_lens, max_mem_len):
    """The attention layer, we use the dot product attention

    Args:
      query: the query vector, [B, S]
      memory: the memory matrix, [B, M, S]
    """
    # We use the scaled dot product attention. Essentially, this method does not
    # include more parameters
    # TODO: other attention methods

    with tf.compat.v1.variable_scope("attention"):
        batch_size = tf.shape(input=memory)[0]
        state_size = tf.cast(tf.shape(input=query)[1], tf.float32)
        query = tf.expand_dims(query, 1)  # [B, 1, S]
        memory_ = tf.transpose(a=memory, perm=[0, 2, 1])  # [B, S, M]
        weights = tf.matmul(query, memory_) / tf.sqrt(state_size)  # [B, 1, M]
        weights = tf.reshape(weights, [batch_size, max_mem_len])

        mem_pad = 1 - tf.sequence_mask(mem_lens, max_mem_len, tf.float32)
        mem_pad *= -10000000
        weights += mem_pad
        dist = tf.nn.softmax(weights)  # [B, M]

        dist_ = tf.expand_dims(dist, 2)  # [B, M, 1]
        context = tf.reduce_sum(input_tensor=dist_ * memory, axis=1)  # [B, S]
    return context, dist


def decoding_infer(start_id,
                   dec_cell,
                   dec_proj,
                   embedding_matrix,
                   enc_state,
                   memory,
                   batch_size,
                   max_dec_len,
                   mem_lens,
                   max_mem_len,
                   state_size=500,
                   multi_source=False,
                   copy=False,
                   copy_ind=None,
                   dec_ptr_g_proj=None,
                   dec_ptr_k_proj=None,
                   bow_cond=None,
                   bow_cond_gate_proj=None):
    """The greedy decoding algorithm, used for inference"""
    dec_outputs = tf.TensorArray(tf.float32, size=max_dec_len)
    dec_out_index = tf.TensorArray(tf.int32, size=max_dec_len)
    start_id = tf.zeros([batch_size], dtype=tf.int32) + start_id
    dec_state = enc_state

    def _dec_loop_attn_fn(i, prev_id, dec_state, dec_outputs, dec_out_index):
        dec_in = tf.nn.embedding_lookup(params=embedding_matrix, ids=prev_id)

        query = dec_state[-1].h
        if (multi_source):
            context, _ = multi_source_attention(
                query, memory, mem_lens, max_mem_len, state_size)
        else:
            context, dist = attention(query, memory, mem_lens, max_mem_len)
        attn_vec = context + query

        bow_cond_g = 1.0
        if (bow_cond_gate_proj is not None):
            bow_cond_g = bow_cond_gate_proj(query + bow_cond)
        if (bow_cond is not None):
            dec_in = dec_in + bow_cond_g * bow_cond

        dec_out, dec_state = dec_cell(dec_in + attn_vec, dec_state)
        dec_outputs = dec_outputs.write(i, dec_out)

        # project to the vocabulary
        dec_logits = dec_proj(dec_out)
        vocab_dist = tf.nn.softmax(dec_logits)

        if (copy):
            print('Using copy mechanism in inference')
            pointers = []
            for proj_i in dec_ptr_k_proj:
                ptr_query = proj_i(dec_out)
                _, ptr = attention(
                    ptr_query, memory[0], mem_lens[0], max_mem_len[0])
                pointers.append(ptr)
            # [num_pointers, batch_size, max_mem_len]
            pointers = tf.stack(pointers)
            pointers = tf.reduce_mean(input_tensor=tf.transpose(a=pointers, perm=[1, 0, 2]), axis=1)

            g = dec_ptr_g_proj(dec_out)
            mixed_dist, ptr_dist = _mix_dist(vocab_dist, pointers, copy_ind, g)
            dec_dist = mixed_dist
        else:
            print('Not using copy mechanism in inference')
            dec_dist = vocab_dist

        dec_index = tf.argmax(input=dec_dist, axis=1, output_type=tf.int32)

        dec_out_index = dec_out_index.write(i, dec_index)
        return (i + 1, dec_index, dec_state, dec_outputs, dec_out_index)

    print("Attention decoding ... ")
    _dec_loop = _dec_loop_attn_fn

    loop_len = max_dec_len
    start_time = 0

    finish_time, _, dec_state, dec_outputs, dec_out_index = tf.while_loop(
        cond=lambda i, _1, _2, _3, _4: tf.less(i, loop_len),
        body=_dec_loop,
        loop_vars=(start_time, start_id, dec_state, dec_outputs, dec_out_index))

    dec_outputs = tf.transpose(a=dec_outputs.stack(), perm=[1, 0, 2])
    dec_out_index = tf.transpose(a=dec_out_index.stack(), perm=[1, 0])
    return dec_outputs, dec_out_index


def _mix_dist(vocab_dist, pointers, memory, g):
    """Given previous decoder state, and current pointers, return the mixed
    distribution.

    Mapping the pointer distribution to the vocabulary distribution by the
    word indices in the memory is a little bit tricky. We implement it by
    the function scatter_nd

    Args:
      vocab_dist: shape = [batch_zize, vocab_size]
      pointers: shape = [batch_size, mem_size]
      memory: shape = [batch_size, max_mem_len]
      g: shape = [batch_size], the mix gate

    Returns:
      mixed_dist: the mixed distribution
      ptr_dist: the pointer distribution
    """

    dshape = tf.shape(input=vocab_dist)

    # from the pointer and the memory, get the new distribution
    mshape = tf.shape(input=memory)

    batch_nums = tf.transpose(a=tf.reshape(tf.tile(
        tf.range(0, mshape[0]), [mshape[1]]), (mshape[1], mshape[0])), perm=[1, 0])
    indices = tf.concat(
        [tf.expand_dims(batch_nums, 2), tf.expand_dims(memory, 2)],
        2)

    ptr_dist = tf.scatter_nd(indices=indices, updates=pointers,
                             shape=dshape)
    mixed_dist = (1 - g) * vocab_dist + g * ptr_dist
    return mixed_dist, ptr_dist


def decoding_train(dec_inputs,
                   dec_cell,
                   dec_proj,
                   enc_state,
                   memory,
                   max_dec_len,
                   mem_lens,
                   max_mem_len,
                   state_size,
                   multi_source=False,
                   copy=False,
                   copy_ind=None,
                   dec_ptr_g_proj=None,
                   dec_ptr_k_proj=None,
                   bow_cond=None,
                   bow_cond_gate_proj=None):
    """The greedy decoding algorithm, used for training"""
    dec_outputs = tf.TensorArray(tf.float32, size=max_dec_len)
    dec_logits_train = tf.TensorArray(tf.float32, size=max_dec_len)
    dec_pointers = tf.TensorArray(tf.float32, size=max_dec_len)
    dec_prob_train = tf.TensorArray(tf.float32, size=max_dec_len)
    dec_g_train = tf.TensorArray(tf.float32, size=max_dec_len)
    dec_inputs = tf.transpose(a=dec_inputs, perm=[1, 0, 2])  # [T, B, S]
    dec_state = enc_state

    if (copy):
        print('Using copy mechanism in decoding')
    else:
        print('Not using copy mechanism')
    if (bow_cond is not None):
        print('Using bow condition vector')
    else:
        print('Not using bow condition vector')
    if (bow_cond_gate_proj is not None):
        print('Using bow condition gate')
    else:
        print('Not using bow condition gate')

    def _dec_loop_attn_fn(i, dec_state,
                          dec_outputs, dec_logits_train, dec_pointers, dec_prob_train, dec_g_train):
        # TODO: the copy mechanism
        dec_in = dec_inputs[i]
        query = dec_state[-1].h
        if (multi_source):
            context, _ = multi_source_attention(
                query, memory, mem_lens, max_mem_len, state_size)
        else:
            context, dist = attention(query, memory, mem_lens, max_mem_len)
        attn_vec = context + query

        bow_cond_g = 1.0
        if (bow_cond_gate_proj is not None):
            bow_cond_g = bow_cond_gate_proj(query + bow_cond)
        if (bow_cond is not None):
            dec_in = dec_in + bow_cond_g * bow_cond

        # dec_out, dec_state = dec_cell(tf.concat([dec_in, attn_vec], 1), dec_state)
        dec_out, dec_state = dec_cell(dec_in + attn_vec, dec_state)
        dec_logits = dec_proj(dec_out)
        vocab_dist = tf.nn.softmax(dec_logits)

        if (copy):
            # the attention source is the
            pointers = []
            print('use %d pointers' % len(dec_ptr_k_proj))
            for proj_i in dec_ptr_k_proj:
                ptr_query = proj_i(dec_out)
                _, ptr = attention(
                    ptr_query, memory[0], mem_lens[0], max_mem_len[0])
                pointers.append(ptr)
            # [num_pointers, batch_size, max_mem_len]
            pointers = tf.stack(pointers)
            pointers = tf.reduce_mean(input_tensor=tf.transpose(a=pointers, perm=[1, 0, 2]), axis=1)

            g = dec_ptr_g_proj(dec_out)

            mixed_dist, ptr_dist = _mix_dist(vocab_dist, pointers, copy_ind, g)

            dec_pointers = dec_pointers.write(i, pointers)
            dec_prob_train = dec_prob_train.write(i, mixed_dist)
            dec_g_train = dec_g_train.write(i, g)

        dec_outputs = dec_outputs.write(i, dec_out)
        dec_logits_train = dec_logits_train.write(i, dec_logits)
        return (i + 1, dec_state,
                dec_outputs, dec_logits_train, dec_pointers, dec_prob_train, dec_g_train)

    print("Attention decoding ... ")
    _dec_loop = _dec_loop_attn_fn

    loop_len = max_dec_len
    start_time = 0

    (finish_time, dec_state, dec_outputs, dec_logits_train, dec_pointers,
     dec_prob_train, dec_g_train) = tf.while_loop(
        cond=lambda i, _1, _2, _3, _4, _5, _6: tf.less(i, loop_len),
        body=_dec_loop,
        loop_vars=(start_time, dec_state,
                   dec_outputs, dec_logits_train, dec_pointers, dec_prob_train, dec_g_train))

    dec_logits_train = tf.transpose(a=dec_logits_train.stack(), perm=[1, 0, 2])
    dec_pointers = tf.transpose(a=dec_pointers.stack(), perm=[1, 0, 2])
    dec_prob_train = tf.transpose(a=dec_prob_train.stack(), perm=[1, 0, 2])
    dec_g_train = tf.transpose(a=dec_g_train.stack(), perm=[1, 0, 2])

    avg_max_ptr = tf.reduce_max(input_tensor=dec_pointers, axis=2)
    pointer_ent = tf.reduce_sum(input_tensor=-dec_pointers *
                                             tf.math.log(dec_pointers + 1e-10), axis=2)
    avg_num_copy = tf.squeeze(dec_g_train)

    return dec_logits_train, dec_prob_train, pointer_ent, avg_max_ptr, avg_num_copy


def decode(dec_start_id,
           dec_inputs,
           dec_cell,
           dec_proj,
           embedding_matrix,
           init_state,
           memory,
           mem_lens,
           max_mem_len,
           batch_size,
           max_dec_len,
           state_size,
           multi_source=False,
           copy=False,
           copy_ind=None,
           dec_ptr_g_proj=None,
           dec_ptr_k_proj=None,
           bow_cond=None,
           bow_cond_gate_proj=None):
    """The decoder, create the decoding graph for both training and inference"""
    # greedy decoding
    _, dec_outputs_predict = decoding_infer(dec_start_id,
                                            dec_cell,
                                            dec_proj,
                                            embedding_matrix,
                                            init_state,
                                            memory,
                                            batch_size,
                                            max_dec_len,
                                            mem_lens,
                                            max_mem_len,
                                            state_size,
                                            multi_source=multi_source,
                                            copy=copy,
                                            copy_ind=copy_ind,
                                            dec_ptr_g_proj=dec_ptr_g_proj,
                                            dec_ptr_k_proj=dec_ptr_k_proj,
                                            bow_cond=bow_cond,
                                            bow_cond_gate_proj=bow_cond_gate_proj)

    dec_logits_train, dec_prob_train, pointer_ent, avg_max_ptr, avg_num_copy = \
        decoding_train(
            dec_inputs,
            dec_cell,
            dec_proj,
            init_state,
            memory,
            max_dec_len,
            mem_lens,
            max_mem_len,
            state_size=state_size,
            multi_source=multi_source,
            copy=copy,
            copy_ind=copy_ind,
            dec_ptr_g_proj=dec_ptr_g_proj,
            dec_ptr_k_proj=dec_ptr_k_proj,
            bow_cond=bow_cond,
            bow_cond_gate_proj=bow_cond_gate_proj)
    return (dec_outputs_predict, dec_logits_train, dec_prob_train, pointer_ent,
            avg_max_ptr, avg_num_copy)