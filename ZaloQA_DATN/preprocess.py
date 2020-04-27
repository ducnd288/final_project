from tqdm import tqdm
from os.path import join, exists
import json
import collections
import random
import math
from transformers import BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from underthesea import sent_tokenize
from underthesea import word_tokenize

random.seed(0)


class InputExample(object):
    """A single training/test example in Zalo format for simple sequence classification."""

    def __init__(self, guid, question, text, title=None, label=None):
        """ Constructs a InputExample.
            :parameter guid: Unique id for the example.
            :parameter question: The untokenized text of the first sequence.
            :parameter text (Optional): The untokenized text of the second sequence
            :parameter label (Optional): The label of the example. This should be
            :parameter title (Optinal): The Wikipedia title where the text is retrieved
        """
        self.guid = guid
        self.question = question
        self.text = text
        self.title = title
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example



class ZaloDatasetProcessor(object):
    """ Base class to process & store input data for the Zalo AI Challenge dataset"""
    

    def __init__(self, dev_size=0.2):
        """ ZaloDatasetProcessor constructor
            :parameter dev_size: The size of the development set taken from the training set
        """
        self.train_data = []
        self.dev_data = []
        self.test_data = []
        self.dev_size = dev_size
        self.label_list = ['False', 'True']

    def load_from_path(self, dataset_path, train_filename, test_filename, dev_filename=None,
                       train_augmented_filename=None, testfile_mode='zalo', encode='utf-8',):
        """ Load data from file & store into memory
            Need to be called before preprocess(before write_all_to_tfrecords) is called
            :parameter dataset_path: The path to the directory where the dataset is stored
            :parameter train_filename: The name of the training file
            :parameter test_filename: The name of the test file
            :parameter dev_filename: The name of the development file
            :parameter train_augmented_filename: The name of the augmented training file
            :parameter testfile_mode: The format of the test dataset (either 'zalo' or 'normal' (same as train set))
            :parameter encode: The encoding of every dataset file
        """
        testfile_mode = testfile_mode.lower()
        assert testfile_mode in ['zalo', 'normal'], "[Preprocess] Test file mode must be 'zalo' or 'normal'"

        def read_to_inputexamples(filepath, encode='utf-8', mode='normal'):
            """ A helper function that read a json file (Zalo-format) & return a list of InputExample
                :parameter filepath The source file path
                :parameter encode The encoding of the source file
                :parameter mode Return data for training ('normal') or for submission ('zalo')
                :returns A list of InputExample for each data instance, order preserved
            """
            try:
                with open(filepath, 'r', encoding=encode) as file:
                    data = json.load(file)
                if mode == 'zalo':
                    returned = []
                    for data_instance in tqdm(data):
                        returned.extend(InputExample(guid=data_instance['__id__'] + '$' + paragraph_instance['id'],
                                                     question=data_instance['question'],
                                                     title=data_instance['title'],
                                                     text=paragraph_instance['text'],
                                                     label=None)
                                        for paragraph_instance in data_instance['paragraphs'])
                    return returned
                else:  # mode == 'normal'
                    return [InputExample(guid=data_instance['id'],
                                         question=data_instance['question'],
                                         title=data_instance['title'],
                                         text=data_instance['text'],
                                         label=self.label_list[data_instance['label']])
                            for data_instance in tqdm(data)]
            except FileNotFoundError:
                return []

        # Get augmented training data (if any), convert to InputExample
        if train_augmented_filename:
            train_data_augmented = read_to_inputexamples(filepath=join(dataset_path, train_augmented_filename),
                                                         encode='utf-8-sig')
            random.shuffle(train_data_augmented)
            self.train_data.extend(train_data_augmented)

        # Get train data, convert to InputExamples
        train_data = []
        if train_filename is not None:
            train_data = read_to_inputexamples(filepath=join(dataset_path, train_filename),
                                               encode=encode)
        # Get dev data, convert to InputExample
        if dev_filename is not None:
            dev_data = read_to_inputexamples(filepath=join(dataset_path, dev_filename),
                                             encode=encode)
            self.dev_data.extend(dev_data)
        # Check if development data exists
        if len(self.dev_data) == 0:
            # Dev data doesn't exists --> Take dev_size of training data
            self.dev_data.extend(train_data[::int(1. / self.dev_size)])  # Get x% of train data evenly
            train_data = [data for data in train_data if data not in self.dev_data]
        self.train_data.extend(train_data)

        # Shuffle training data
        random.shuffle(self.train_data)

        # Get test data, convert to InputExample
        if test_filename is not None:
            test_data = read_to_inputexamples(filepath=join(dataset_path, test_filename),
                                              encode=encode, mode=testfile_mode)
            self.test_data.extend(test_data)

    def _convert_single_example(self, example, label_list, max_seq_length, tokenizer):
        """ Converts a single `InputExample` into a single `InputFeatures`.
            :parameter example: A InputRecord instance represent a data instance
            :parameter label_list: List of possible labels for predicting
            :parameter max_seq_length: The maximum input sequence length for embedding
            :parameter tokenizer: A BERT-based tokenier to tokenize text
        """
        # Labels mapping
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        # Text tokenization
    #    tokens_a = tokenizer.tokenize(example.question)
    #    tokens_b = None
    #    if example.text:
    #        tokens_b = tokenizer.tokenize(example.text)

        def _truncate_seq_pair(ques, text, max_length):
            """Truncates a sequence pair in place to the maximum length."""
            ques_1 = ques
            sens = sent_tokenize(text)
            sens_t = []
            sen_tokens = []
            ques_tokens = tokenizer.tokenize(ques_1)
            for sen in sens:
                tokens = tokenizer.tokenize(sen)
                sen_tokens.append(tokens)
                sens_t.append(' '.join(tokens))
            ques_in = ' '.join(ques_tokens)

            def ranking_ques_sentences(ques, sentences):
                corpus = [ques]
                corpus.extend(sentences)
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(corpus)
                X = X.toarray()
                ques = np.array([(len(X) - 1) * X[0]])
                sens = np.array(X[1:])
                rank_list = cosine_similarity(ques, sens).reshape(-1).tolist()
                return sorted(range(len(rank_list)), key=lambda k: rank_list[k])
                

            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            rl = None
            i = 0
            while True:
                total_length = sum([len(sen) for sen in sen_tokens]) + len(ques_tokens)
                if total_length <= max_length:
                    break
                else:
                    if i==0:
                        rl = ranking_ques_sentences(ques_in, sens_t)
                    try:
                        sen_tokens[rl[i]] = []
                        i = i + 1
                    except IndexError:
                        return None, None
                    
            tokens_b = []
            for sen_ in sen_tokens:
                if len(sen_)!=0:
                    tokens_b += sen_
            
            return ques_tokens, tokens_b 


        tokens_a = None
        tokens_b = None
        # Truncate text if total length of combinec input > max sequence length for the model
        if example.text:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = _truncate_seq_pair(example.question, example.text, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            tokens_a = tokenizer.tokenize(example.question)
            
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
        

        if tokens_a is None:
            return None

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label] if example.label is not None else -1

        feature = InputFeatures(
            guid=example.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """ Convert a set of `InputExample`s to a list of `InputFeatures`. (Helper class for prediction)
            :parameter examples: List of InputRecord instances that need to be processed
            :parameter label_list: List of possible labels for predicting
            :parameter max_seq_length: The maximum input sequence length for embedding
            :parameter tokenizer: A BERT-based tokenier to tokenize text
        """
        features = []
        for (ex_index, example) in enumerate(examples):
            feature = self._convert_single_example(example, label_list, max_seq_length, tokenizer)
            if feature is not None:
                features.append(feature)
        return features

    