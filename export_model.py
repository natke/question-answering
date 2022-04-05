import torch
import onnxruntime
from transformers import BertTokenizer, BertForQuestionAnswering

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
output_model_path = "./" + model_name + ".onnx"

model = BertForQuestionAnswering.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained(model_name)

question = "What is my name?"
context = "My name is Natalie"
input_ids = tokenizer.encode(question, context)
print("The input has a total of {} tokens.".format(len(input_ids)))

tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):
    print('{:8}{:8,}'.format(token,id))

 # First occurence of [SEP] token
sep_idx = input_ids.index(tokenizer.sep_token_id)
print("SEP token index: ", sep_idx)#number of tokens in segment A (question) - this will be one more than the sep_idx as the index in Python starts from 0
num_seg_a = sep_idx+1
print("Number of tokens in segment A: ", num_seg_a)#number of tokens in segment B (text)
num_seg_b = len(input_ids) - num_seg_a
print("Number of tokens in segment B: ", num_seg_b)#creating the segment ids
segment_ids = [0]*num_seg_a + [1]*num_seg_b#making sure that every input token has a segment id
assert len(segment_ids) == len(input_ids)

device = 'cpu'

# Run the torch model as a test
output = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids])) 

# set the model to inference mode
# It is important to call torch_model.eval() or torch_model.train(False) before exporting the model, 
# to turn the model to inference mode. This is required since operators like dropout or batchnorm 
# behave differently in inference and training mode.
model.eval()

# Generate dummy inputs to the model. Adjust if neccessary
inputs = {
        'input_ids':   torch.randint(32, [1, 32], dtype=torch.long).to(device), # list of numerical ids for the tokenized text
        'token_type_ids': torch.ones([1, 32], dtype=torch.long).to(device),     # dummy list of ones
    }

symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
torch.onnx.export(model,                                        # model being run
                  (inputs['input_ids'], 
                   inputs['token_type_ids']),                   # model input (or a tuple for multiple inputs)
                  output_model_path,                            # where to save the model (can be a file or file-like object)
                  opset_version=11,                             # the ONNX version to export the model to
                  do_constant_folding=True,                     # whether to execute constant folding for optimization
                  input_names=['input_ids', 
                               'segment_ids'],                  # the model's input names
                  output_names=['start', "end"],                # the model's output names
                  dynamic_axes={'input_ids': symbolic_names,              
                                'segment_ids' : symbolic_names,
                                'start' : symbolic_names, 
                                'end': symbolic_names})         # variable length axes



