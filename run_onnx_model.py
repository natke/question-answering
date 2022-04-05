import onnxruntime
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
output_model_path = "./" + model_name + ".onnx"

model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

question = "What is my name?"
context = "My name is Natalie"
input_ids = tokenizer.encode(question, context)

tokens = tokenizer.convert_ids_to_tokens(input_ids)
#for token, id in zip(tokens, input_ids):
#    print('{:8}{:8,}'.format(token,id))

# First occurance of [SEP] token
sep_idx = input_ids.index(tokenizer.sep_token_id)
print("SEP token index: ", sep_idx)
num_seg_a = sep_idx+1
print("Number of tokens in the question: ", num_seg_a)
num_seg_b = len(input_ids) - num_seg_a
print("Number of tokens in the context: ", num_seg_b)
segment_ids = [0]*num_seg_a + [1]*num_seg_b
assert len(segment_ids) == len(input_ids)

# Run the torch model as a test
output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])) 

# Create an ONNX Runtime session to run the ONNX model
session = onnxruntime.InferenceSession(output_model_path)  

inputs = {
        'input_ids':   [input_ids], 
        'segment_ids': [segment_ids]
        }
                    
result = session.run(["start", "end"], inputs)

#tokens with highest start and end scores
answer_start = torch.argmax(torch.from_numpy(result[0]))
answer_end = torch.argmax(torch.from_numpy(result[1]))
if answer_end >= answer_start:
    answer = tokens[answer_start]
    for i in range(answer_start+1, answer_end+1):
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]
        else:
            answer += " " + tokens[i]
    print("\nQuestion:\n{}".format(question.capitalize()))
    print("\nAnswer:\n{}.".format(answer.capitalize()))
else:
    print("I am unable to find the answer to this question. Can you please ask another question?")
    
