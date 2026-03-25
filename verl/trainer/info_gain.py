import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import gc

class InfoGainAdvantageCalculator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化 Embedding 模型
        """
        
        # 优化1：以 bfloat16 加载模型，显存占用减半
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            trust_remote_code=True,
            device="cpu"
        )
        # 优化2：默认放在 CPU 上，用的时候再搬到 GPU
        self.model.eval()

    def get_embeddings(self, texts):
        """
        获取文本的 Embedding 表示 (分块处理防 OOM)
        """ 
        # 限制 CPU 线程数，防止占用所有核心导致 Ray 心跳超时 (ActorDiedError)
        torch.set_num_threads(4)
        
        # 优化3：用的时候再搬到 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        
        # 优化4：直接使用 sentence-transformers 的 encode 方法
        # 它会自动处理 batching, tokenization, pooling 等
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=4,               # 极小 batch_size 防止 GPU OOM
                normalize_embeddings=False, # 后续统一做 normalize
                convert_to_tensor=True,     # 确保返回的是 PyTorch Tensor
                device=device               # 确保在 GPU 上运行
            )
            
        # 优化7：将 embedding 移到 CPU，后续的最小二乘法在 CPU 上算，彻底解放 GPU 显存
        embeddings = embeddings.to(torch.float32).cpu()
            
        # 优化6：用完立刻把模型踢回 CPU 并清空缓存
        self.model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        
        return embeddings

def distance(a, b):
    # Use the last dimension so both 1D and batched tensors are supported.
    dis = (1 - torch.cosine_similarity(a, b, dim=-1)) / 2
    return dis

    
