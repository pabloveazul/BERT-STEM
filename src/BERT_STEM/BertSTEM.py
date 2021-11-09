import torch
import numpy as np
import pandas as pd
import transformers

from BERT_STEM.Encode import *
from BERT_STEM.TrainBERT4PT import *

class BertSTEM:
    
    def __init__(self, model_name = None):
        
        if model_name is not None:
            self.model_name = model_name

        else:
            self.model_name = "pablouribe/bertstem"

        self._model = transformers.BertModel.from_pretrained(self.model_name)

        self._tokenizer = transformers.BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased",
                                                                         do_lower_case = True,
                                                                         add_special_tokens = False)

    def _encode_df(self, df, column, encoding):

        return(sentence_encoder(df, self._model, self._tokenizer, column, encoding))

    def get_embedding_matrix(self):

        #Build the Embedding Matrix
        max_vocab = 50105
        bert_dim = 768
        embedding_matrix = np.zeros((max_vocab, bert_dim))
        
        with torch.no_grad():
            for i in range(max_vocab):
                try:
                    tokenized_text = self._tokenizer.decode([i])
                    indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens]).to(torch.int64)
                    encoded_layers = self._model.get_input_embeddings()(tokens_tensor)[0]
                    
                    embedding_vector = np.array(encoded_layers)
                    embedding_matrix[i] = embedding_vector
                except:
                    pass

        return(embedding_matrix)


class BertSTEMForPreTraining:
    
    def __init__(self, model_name = None):
        
        if model_name is not None:
            self.model_name = model_name
        
        else:
            self.model_name = "pablouribe/bertstem"
        
        self._model = transformers.BertForPreTraining.from_pretrained(self.model_name)
        
        self._tokenizer = transformers.BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased",
                                                                         do_lower_case = True,
                                                                         add_special_tokens = False)

    def _train_bert_for_pretraining(self, text, checkpoint):
            
        train_Bert4PT(self._model, self._tokenizer, text)
                
    def _get_text_from_files(self, files_path):
            
            return(get_text_from_files(files_path))


class BertSTEMForTextClassification:
    
    def __init__(self, model_name = None):
        
        if model_name is not None:
            self.model_name = model_name
        
        else:
            self.model_name = "pablouribe/bertstem"
        
        self._model = transformers.BertForSequenceClassification.from_pretrained(self.model_name)
        
        self._tokenizer = transformers.BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased",
                                                                         do_lower_case = True,
                                                                         add_special_tokens = False)

