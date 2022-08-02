from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import anltk
import unicodedata
import torch

# Guide dependencies
from abc import ABC
import logging
import torch
import kserve
import argparse
# from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class TopicClassifier(kserve.Model):
    def __init__(self,):
        super().__init__()
        logger.log(logging.INFO, "=============INITIALIZING TOPIC CLASSIFIER=============")
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_path = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, config=self.config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)

        self.model.to(self.device)
        self.model.eval()

        # Optional Mapping File
        self.labels = list(self.config.label2id.keys())


        logger.log(logging.INFO, f"Initialized Topic Classifier")
        self.initialized = True

    def preprocess_(self, query: str) -> str:
        query_ = unicodedata.normalize('NFC', query)
        query_ = ' '.join(anltk.tokenize_words(query_))
        query_ = anltk.remove_non_alpha(query_, stop_list=' ?,:".')
        query_ = anltk.fold_white_spaces(query_)
        query_ = query_.strip()
        return query_

    def preprocess(self, data):
        query = data[0].get("data")
        if query is None:
            query = data[0].get("body")

        query_ = self.preprocess(query)

        # TODO: These are used only for display, remove in production
        tokens = self.tokenizer.tokenize(query_)

        encoded_dict = self.tokenizer.encode_plus(
                        query_,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length=512,           # Pad & truncate all sentences.
                        truncation=True,
                        padding='max_length',  # Padding strategy
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
        return encoded_dict, tokens, query_
    
    def inference(self, inputs,):
        logger.log(logging.INFO, f"Inference started")
        with torch.no_grad():
            for key in inputs: # Convert all to device first
                try: 
                    inputs[key] = inputs[key].to(self.device)
                except:
                    pass
            outputs = self.model(**inputs)
        
        predictions = torch.nn.functional.softmax(outputs[0].squeeze(), dim=0)
        pred = torch.argmax(predictions, dim=0)

        correct = self.labels[pred.item()]

        logger.log(logging.INFO, f"Predicted: {correct}")

        labeled_dict = {"Correct": correct}
        for label in self.labels:
            labeled_dict[label] = "{:.3f}".format(predictions[self.config.label2id[label]].item())
        
        return labeled_dict

    def postprocess(self, data: dict, tokens, query):
        data["Tokens"] = tokens
        data["Preprocessed"] = query
        return [data] # Return the data as is but in a list

_service = TopicClassifier()

def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)
        
        if data is None:
            return None
        
        inputs, tokens, query = _service.preprocess(data)
        output_dict = _service.inference(inputs)
        outputs = _service.postprocess(output_dict, tokens, query)

        return outputs
    except Exception as e:
        logger.error(e)
        raise e

