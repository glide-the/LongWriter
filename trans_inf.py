from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("/mnt/ceph/develop/jiawei/LongWriter-glm4-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/ceph/develop/jiawei/LongWriter-glm4-9b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = model.eval()
query = "请帮我撰写一个主题为「黑神话·悟空」玄幻小说，小说以孙悟空为核心，讲述一个桀骜不驯，打怪升级，追逐梦想的玄幻故事，不少于 10000 字"
response, history = model.chat(tokenizer, query, history=[], max_new_tokens=32768, temperature=0.5)
print(response)