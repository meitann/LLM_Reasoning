from transformers import AutoTokenizer
import torch

def compute_token_lengths(texts, tokenizer):
    encodings = tokenizer(texts)
    token_lengths = [len(encoding['input_ids']) for encoding in encodings['input_ids']]
    return token_lengths

def compress(v, d):
    original_dim = v.shape[1]
    
    group_size = original_dim // d
    
    compressed_v = []
    for i in range(d):
        start = i * group_size
        end = (i + 1) * group_size
        group = v[:, start:end]
        group_sum = torch.sum(group, dim=1)
        normalized_group = group_sum / torch.sqrt(torch.tensor(group_size, dtype=torch.float32)) 
        
        compressed_v.append(normalized_group.unsqueeze(1))
    compressed_v = torch.cat(compressed_v, dim=1)
    
    return compressed_v