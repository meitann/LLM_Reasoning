# 首先下载一个预训练模型
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("D:/math/A-reasoning_demo/Models/LaBSE")
model = SentenceTransformer("sentence-transformers/LaBSE",device='cuda')
# 然后提供一些句子给模型
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
 
# 现在有了一个带有嵌入的NumPy数组列表
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")








# from transformers import AutoModel, AutoTokenizer
# import torch
# # 加载模型和分词器
# model = AutoModel.from_pretrained(r"D:\math\A-reasoning_demo\Models\LaBSE")
# tokenizer = AutoTokenizer.from_pretrained(r"D:\math\A-reasoning_demo\Models\LaBSE")
# sentences = ['This framework generates embeddings for each input sentence', 'Sentences are passed as a list of string.']
# inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# with torch.no_grad():
#     embeddings = model(**inputs).last_hidden_state.mean(dim=1)

# print(embeddings.shape)