from preprocess import *
from training import Training
from transformers import *
from modeling import *
from utils import init_logger
import logging
import torch

torch.cuda.manual_seed_all(2018)
torch.manual_seed(2018)
torch.backends.cudnn.deterministic = True
random.seed(2018)

init_logger()
logger = logging.getLogger(__name__)

config = {}
config['batch_size'] = 16
config['epochs'] = 5
config['lr'] = 1e-5
config['accumulated'] = 5
fine_tuning = True


zalo = ZaloDatasetProcessor()
zalo.load_from_path(dataset_path='dataset', train_filename='train_dev.json', test_filename='test.json', dev_filename='dev.json')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
features_train = zalo.convert_examples_to_features(zalo.train_data, zalo.label_list, 256, tokenizer)
features_test = zalo.convert_examples_to_features(zalo.test_data, zalo.label_list, 256, tokenizer)
features_dev = zalo.convert_examples_to_features(zalo.dev_data, zalo.label_list, 256, tokenizer)
if __name__ == "__main__":
    
    NUM_OF_INTENT = 2
    config_model = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
    model = QAModel(config_model, NUM_OF_INTENT)
    if fine_tuning:
      model.load_state_dict(torch.load('models/model-squad1.bin'))
    training = Training(features_train, features_dev, model, logger, zalo.label_list, config)
    training.train()