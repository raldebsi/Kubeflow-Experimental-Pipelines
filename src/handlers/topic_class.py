from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import anltk
import unicodedata
import torch

# Guide dependencies
from abc import ABC
import logging
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class TopicClassifier(BaseHandler, ABC):
    def __init__(self,):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        logger.log(logging.INFO, "=============INITIALIZING TOPIC CLASSIFIER=============")
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_path = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, config=self.config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)

        self.model.to(self.device)
        self.model.eval()

        # Label Mapping
        self.labels = list(self.config.label2id.keys())


        logger.log(logging.INFO, f"Initialized Topic Classifier")
        self.initialized = True

    def preprocess_(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""
        query_ = unicodedata.normalize('NFC', query)
        query_ = ' '.join(anltk.tokenize_words(query_))
        query_ = anltk.remove_non_alpha(query_, stop_list=' ?,:".')
        query_ = anltk.fold_white_spaces(query_)
        query_ = query_.strip()
        return query_

    def preprocess(self, data):
        logger.log(logging.INFO, f"Preprocessing started")
        logger.log(logging.INFO, f"data is {data}")
        query = data[0]
        query = query.get("body", {"text": query.get("text", "")}).get("text", "")
        query = self.preprocess_(query)
        if not query.strip():
            raise Exception("No text found in query")
        logger.log(logging.INFO, f"query is {query}")

        query_ = self.preprocess_(query)

        # tokens = self.tokenizer.tokenize(query_) # Debugging only

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
        return (encoded_dict, query_)
    
    def inference(self, inputs,):
        with torch.no_grad():
            for key, val in inputs.items(): # Convert all to device first
                try: 
                    inputs[key] = val.to(self.device)
                except:
                    pass
        outputs = self.model(**inputs)
        
        predictions = torch.nn.functional.softmax(outputs[0].squeeze(), dim=0)
        pred = torch.argmax(predictions, dim=0)

        correct = self.labels[pred.item()]

        logger.log(logging.INFO, f"Predicted: {correct}")

        class_dict = {}
        labeled_dict = {"Correct": correct, "Classes": class_dict}
        for label in self.labels:
            class_dict[label] = "{:.3f}".format(predictions[self.config.label2id[label]].item())
        
        return labeled_dict

    def postprocess(self, data: dict, query):
        # data["Preprocessed"] = query # No Need
        return [data] # Return the data as is but in a list

_service = TopicClassifier()

logger.log(logging.INFO, f"{__name__} is Running")

def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)
        
        if data is None:
            return None
        
        logger.log(logging.INFO, f"Received data: {data}")
        
        inputs, query = _service.preprocess(data)
        output_dict = _service.inference(inputs)
        outputs = _service.postprocess(output_dict, query)

        return outputs
    except Exception as e:
        logger.error(e)
        raise e