class Parameters:
    def __init__(self, max_word_len, max_seq_len, word_vocab_size, char_vocab_size):

        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(max_seq_len) + 1 # go or eos token

        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)

        self.word_embed_size = 300
        self.char_embed_size = 15

        self.kernels = [(1, 25), (2, 50), (3, 75), (4, 100), (5, 125), (6, 150)]

        self.encoder_rnn_size = [800, 950, 1100]
        self.encoder_num_rnn_layers = len(self.encoder_rnn_size)

        self.latent_variable_size = 1200

        self.decoder_rnn_size = [1200, 1350, 1400]
        self.decoder_num_rnn_layers = len(self.decoder_rnn_size)