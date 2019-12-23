class Config:

    # training hyperparameters
    batch_size = 100  # 60 for the seq2seq model, effective batch size = 100
    start_epoch = 0
    num_epoch = 100
    train_print_interval = 500

    # dataset setting
    dataset = "quora"
    dataset_path = {
        "quora": {
            "train": "../data/quora.txt",
            "test": ""
        }
    }
    train_index_path = "../data/quora_train_index.txt"
    dev_index_path = "../data/quora_dev_index.txt"

    # pickle
    read_from_pickle = True
    word2id_path = "../data/word2id.pickle"
    id2word_path = "../data/id2word.pickle"
    pos2id_path = "../data/pos2id.pickle"
    id2pos_path = "../data/id2pos.pickle"
    ner2id_path = "../data/ner2id.pickle"
    id2ner_path = "../data/id2ner.pickle"
    train_sentences_path = "../data/train_sentences.pickle"
    train_lens_path = "../data/train_lens.pickle"
    dev_sentences_path = "../data/dev_sentences.pickle"
    dev_lens_path = "../data/dev_lens.pickle"

    max_sent_len = {"quora": 20}  # 95 percentile

    save_ckpt = True

    vocab_size = -1

    dec_start_id = 0
    dec_end_id = 1
    pad_id = 2
    unk_id = 3
    stop_words = None

    ## Model configuration
    model_name = "latent_bow"
    model_mode = "train"  # ["train", "test"]
    model_version = "0.1"
    model_path = "../model/"
    output_path = "../output/"

    state_size = 500
    drop_out = 0.6

    # encoder
    num_paraphrase = 1  # 1 for quora, 4 for mscoco
    enc_layers = 2
    lambda_enc_loss = 1.0
    max_enc_bow = 11  # The number of bag of words, 25 for mscoco, 11 for quora
    no_residual = False

    # decoder
    decoding_mode = "greedy"
    dec_layers = 2
    max_dec_bow = 30  # 10 for mscoco, 11 for quora
    source_sample_ratio = 0.
    sample_size = 30  # 12 for mscoco, 11 for quora, 30 for wikibio

    is_cheat = False
    num_pointers = 3
    bow_cond = False
    bow_cond_gate = False

    ## Controller configuration
    # system setting
    gpu_id = "0"
    controller_mode = "train"

    # evaluation metrics
    eval_metrics_list = ["bleu"]
    log_metrics = ["predict_average_confident", "target_average"]
    write_output = True
    single_ref = False
    compare_outputs = True

    # optimizer
    learning_rate_decay = False
    random_seed = 15213
    target_metrics = "bleu_2"  # ["ppl", "bleu_2"]
    optimizer = "Adam"
    learning_rate = 0.0008
    learning_rate_enc = 0.001  # or 1e-3, sensitive
    learning_rate_dec = 0.001


config = Config()
