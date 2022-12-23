import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from textblob import TextBlob
# !pip install contractions
import contractions
from copy import deepcopy
import torch


class DTPreprocessor():
    """
            This class is used to preprocess a corpus (a list of 
        pairs of strings (texts)) for training a date translator.
        The cleaning process includes
            (1) Lower the case of all characters
            (2) turn multple spaces into one space
            The vocabulary and sequences are constructed as follows:
            (1) For human readable texts check if the training data 
        covers all integers and characters for months and weekdays 
        (Note that this is a loose check since even if it covers all 
        the characters it does not necessarily cover all characters of 
        months and weekdays.)
            (2) Build the vocab for human and machine readable date
        separately. For human readable date, the vocab includes all
        english letters, all integer from 0 to 9 and specified 
        separators. All other characters are also included but are 
        only seen as UNK. For machine readable date, the vocab includes all integer from 0 to 9
    """
    
    INTEGERS = {str(num) for num in range(10)}
    EN_ALPHABET = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
    LEAST_ALPHABET = {*'January' + 'February' + 'March' + 'April' + 'May' + 'June' + 'July' + 'August' + 'September' + 'October' + 'November' + 'December'}
    LEAST_ALPHABET = {i.lower() for i in LEAST_ALPHABET}
    
    def __init__(self, machine_separator_to_use = '-', human_separators_to_keep = '', front_padded = False):
        self.machine_separator_to_use = machine_separator_to_use
        self.human_separators_to_keep = {*human_separators_to_keep}
        self.HUMAN_LEAST_LIB = self.LEAST_ALPHABET.union(self.INTEGERS)
        self.HUMAN_LIB = self.INTEGERS.union(self.EN_ALPHABET).union(self.human_separators_to_keep)
        self.human_tokenized_corpus = list()
        self.machine_tokenized_corpus = list()
        self.human_char2idx = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.human_idx2char = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.machine_char2idx = {"PAD": 0, "SOS": 1, "EOS": 2, self.machine_separator_to_use: 3}
        self.machine_idx2char = {0: "PAD", 1: "SOS", 2: "EOS", 3: self.machine_separator_to_use}
        self.human_sorted_vocab = dict()
        self.machine_sorted_vocab = dict()
        self.human_sequences = list()
        self.machine_sequences = list()
        self.human_num_characters = int()
        self.machine_num_characters = int()
        self.human_num_tokens = int()
        self.machine_num_tokens = int()
        self.front_padded = front_padded
        
    def cleanse_corpus(self, corpus):
        """
            Only the human readable corpus needs to be cleaned.
            Cleaning includes:
                1. Lowercase all letters
                2. turn multple spaces into one space
        """
        
        corpus = pd.DataFrame(corpus)
        
        corpus[0] = corpus[0].map(lambda x: self.format_string(x))
        
        return corpus
        
    def format_string(self, text):
        return ' '.join(text.lower().split())
    
    def fit_on_corpus(self, corpus):
        self.human_tokenized_corpus = self.tokenize_corpus(corpus.iloc[:, 0])
        self.machine_tokenized_corpus = self.tokenize_corpus(corpus.iloc[:, 1])
        
        assert self.test_coverage(self.human_tokenized_corpus, self.HUMAN_LEAST_LIB) and self.test_coverage(self.machine_tokenized_corpus, self.INTEGERS), "Corpus does not cover all possible characters!"
        
        self.human_sorted_vocab, self.human_num_characters = self.build_sorted_vocab(self.human_tokenized_corpus, self.HUMAN_LIB)
        self.machine_sorted_vocab, self.machine_num_characters  = self.build_sorted_vocab(self.machine_tokenized_corpus, self.INTEGERS)
        
        self.human_char2idx.update(self.build_char2idx(self.human_sorted_vocab, self.human_num_characters))
        self.human_idx2char.update(self.build_idx2char(self.human_sorted_vocab, self.human_num_characters))     
        self.machine_char2idx.update(self.build_char2idx(self.machine_sorted_vocab, self.machine_num_characters))
        self.machine_idx2char.update(self.build_idx2char(self.machine_sorted_vocab, self.machine_num_characters))
        
        self.human_num_tokens = len(self.human_char2idx)
        self.machine_num_tokens = len(self.machine_char2idx)

        self.human_sequences = self.texts_to_sequences(self.human_tokenized_corpus, self.human_char2idx, self.HUMAN_LIB, "UNK")
        self.machine_sequences = self.texts_to_sequences(self.machine_tokenized_corpus, self.machine_char2idx, self.INTEGERS, self.machine_separator_to_use)
  
    def test_coverage(self, tokenized_corpus, LIB):
        coverage_dict = {i: False for i in LIB}
        for text in tokenized_corpus:
            for token in text:
                if token in LIB:
                    coverage_dict[token] = True
                else:
                    pass
        return all(coverage_dict.values())
    
    def tokenize_corpus(self, corpus):
        return [[*string] for string in corpus]
    
    def build_sorted_vocab(self, tokenized_corpus, LIB):
        """
            Build a sorted vocabulary containing all characters.
            Note that only the vocab for human readable texts 
            contains UNK
        """
        # 
        vocab = defaultdict(int)
        for text in tokenized_corpus:
            for token in text:
                # If token is in LIB include it in the vocab otherwise ignore it 
                # and it will be treated as UNK for human vocab or separator for
                # machine vocab
                if token in LIB:
                    vocab[token] += 1
                else:
                    pass
        sorted_vocab = list(vocab.items())
        sorted_vocab.sort(key=lambda x: x[1], reverse=True)
                
        return sorted_vocab, len(sorted_vocab)
        
    def build_char2idx(self, sorted_vocab, num_characters):
        # Construct a character -> index mapping for all characters plus the 4 special tokens
        character_list = []
        character_list.extend(item[0] for item in sorted_vocab)
        return dict( zip(character_list, list(range(4, num_characters + 4))) )

    def build_idx2char(self, sorted_vocab, num_characters):
        # Construct a index -> character mapping for all characters plus the 4 special tokens
        character_list = []
        character_list.extend(item[0] for item in sorted_vocab)
        return dict( zip(list(range(4, num_characters + 4)), character_list) )
    
    def texts_to_sequences(self, tokenized_corpus, char2idx, LIB, OOLIB_name):
        return list(self.texts_to_sequences_generator(tokenized_corpus, char2idx, LIB, OOLIB_name))

    def texts_to_sequences_generator(self, tokenized_corpus, char2idx, LIB, OOLIB_name):
        for character_seq in tokenized_corpus:
            seq = list()
            last_token_index = 3
            for token in character_seq:
                # If token is not in LIB, treat it as UNK for human 
                # vocab or separator for machine vocab
                if token in LIB:
                    pass
                else:
                    token = OOLIB_name
                idx = char2idx.get(token)
                # If idx is None (there is no such character in the vocab),
                # skip it (it won't be seen at all)
                if idx is not None and (idx != 3 or idx != last_token_index):
                    seq.append(idx)
                    last_token_index = idx
                else:
                    pass
            seq = [1] + seq + [2]
            yield seq
            
    def pad_minibatch_collate(self, sequences):
        """
                This function takes a torch.tensor and outputs
            a torch.tensor. This is designed to facilitate the 
            construction of a collate_fn for a DataLoader.
                Find the length of the longest sequence in a
            minibatch (either inputs or outputs, a list of 
            sequences) and pad all sequences to that length.
        """
        padded_sequences = deepcopy(list(sequences))
        assert len(padded_sequences[0].shape) == 1, "Incorrect dimension!"
        target_len = max([len(x) for x in padded_sequences])
        for index in range(len(padded_sequences)):
            if len(padded_sequences[index]) < target_len:
                if self.front_padded:
                    padded_sequences[index] = torch.tensor([0 for i in range(target_len - len(padded_sequences[index]))] + list(padded_sequences[index]))
                else:
                    padded_sequences[index] = torch.tensor(list(padded_sequences[index]) + [0 for i in range(target_len - len(padded_sequences[index]))])
            else:
                padded_sequences[index] = padded_sequences[index][:target_len]
        return torch.stack(padded_sequences)
            
    def get_sequences_of_test_texts(self, texts):
        """
                Turn test texts (a list of pairs of strings) into a pair of
                sequences (a pair of lists of sequences).
        """
        texts = pd.DataFrame(texts)
        assert len(texts.columns) > 1, "Input needs to be pairs of strings"
        input_tokenized_texts = self.tokenize_corpus(texts.iloc[:, 0])
        input_seqs = self.texts_to_sequences(input_tokenized_texts, self.human_char2idx, self.HUMAN_LIB, "UNK")
        output_tokenized_texts = self.tokenize_corpus(texts.iloc[:, 1])
        output_seqs = self.texts_to_sequences(output_tokenized_texts, self.machine_char2idx, self.INTEGERS, self.machine_separator_to_use)
        return input_seqs, output_seqs

    def get_sequences_of_inference_texts(self, texts):
        """
                Turn texts for inference (list of strings) into sequences 
            (list of sequences).
        """
        texts = pd.DataFrame(texts)
        tokenized_texts = self.tokenize_corpus(texts.iloc[:, 0])
        seqs = self.texts_to_sequences(tokenized_texts, self.human_char2idx, self.HUMAN_LIB, "UNK")
        return seqs


class DTDataset(torch.utils.data.Dataset):
	def __init__(self, X, y):
		super().__init__()
		self.X = X
		self.y = y

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return (torch.tensor(self.X[idx], dtype = torch.int64), torch.tensor(self.y[idx], dtype = torch.int64))










