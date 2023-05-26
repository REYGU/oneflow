import oneflow as flow
from oneflow import nn
import os
import numpy as np
import time

# class BERT(nn.Module):
#     def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, max_seq_len):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
        
#         # Embedding
#         self.token_embedding = nn.Embedding(vocab_size, hidden_size)
#         self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        
#         # Encoder
#         encoder_layers = []
#         for i in range(num_hidden_layers):
#             encoder_layers.append(self.encoder_layer(hidden_size, 
#                                                     num_attention_heads, 
#                                                     hidden_size))
#         self.encoder = nn.Sequential(*encoder_layers)
        
#         # Pooling
#         self.pooler = nn.Linear(hidden_size, hidden_size)
        
#     def encoder_layer(self, hidden_size, num_attention_heads, ff_dim):
#         return nn.TransformerEncoderLayer(hidden_size, num_attention_heads, ff_dim)
        
#     def forward(self, token_ids, pos_ids):
#         # Embedding
#         embeds = self.token_embedding(token_ids) + self.pos_embedding(pos_ids)
        
#         # Encoder
#         encoder_out = self.encoder(embeds)
        
#         # Pooling
#         first_token = encoder_out[:, 0]
#         pooled = self.pooler(first_token) 
        
#         return pooled, encoder_out



class _TestModuleDiffHierarchy(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x, y):
        o = flow.pow(x, y)
        o = flow.mul(x, y) + flow.mul(x, y) 
        o = flow.div(flow.mul(x, y), flow.div(x, y))
        o = flow.div(x, y)
        # o1 = flow.add(x, y)
        # o2 = flow.div(x, y)
        # o3 = flow.pow(x, y)
        # o = flow.mm(x, y)
        o = flow.nn.functional.relu(o)
        o = flow.softmax(o)
        o = flow.nn.functional.log_softmax(o)
        o = flow.asin(o)
        o = flow.negative(o)
        o = flow.nn.functional.celu(o)
        x = flow.ones_like(o)
        return o


class _TestGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config.enable_auto_parallel(True)

    def build(self, x, y):
        o = self.model(x, y)
        return o


def test_lazy_boxing_2d_all_combination():
    a = flow.rand(131072//4,1024, device="cuda:0")
    b = flow.rand(131072//4,1024, device="cuda:0")
    flow.boxing.nccl.enable_use_compute_stream(True)

   
    model_diff_hierarchy = _TestModuleDiffHierarchy()
    graph_diff_hierarchy = _TestGraph(model_diff_hierarchy)
    print("Start to run graph_diff_hierarchy")
    # record the start time in python
    t1 = time.perf_counter()
    with flow.no_grad():
        y = graph_diff_hierarchy(a, b)
    t2 = time.perf_counter()
    print("time used: ", t2 - t1)


if __name__ == "__main__":
    test_lazy_boxing_2d_all_combination()
