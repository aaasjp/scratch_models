import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
import numpy as np


class SPLADEModel(nn.Module):
    """
    SPLADE (Sparse Lexical and Expansion Model) 实现
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        # 加载预训练BERT模型
        self.transformer = AutoModel.from_pretrained(model_name)
        self.vocab_size = self.transformer.config.vocab_size
        hidden_dim = self.transformer.config.hidden_size
        
        # 线性投影层：将隐藏状态映射到词汇表空间
        self.mlm_head = nn.Linear(hidden_dim, self.vocab_size)
        
        # 初始化投影层权重
        self.mlm_head.weight.data.normal_(mean=0.0, std=0.02)
        if self.mlm_head.bias is not None:
            self.mlm_head.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            sparse_vector: [batch_size, vocab_size] 稀疏表示向量
        """
        # 1. BERT编码
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # 2. 投影到词汇表空间
        logits = self.mlm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        # 3. SPLADE激活函数: log(1 + ReLU(x))
        activated = torch.log1p(F.relu(logits))  # log(1+x) 数值稳定
        
        # 4. Max Pooling: 对序列维度取最大值
        # 使用attention_mask确保padding位置不参与计算
        mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        activated = activated * mask  # 屏蔽padding
        
        # 取每个词汇维度的最大值
        sparse_vector, _ = torch.max(activated, dim=1)  # [batch, vocab_size]
        
        return sparse_vector
    
    def encode(self, texts: List[str], tokenizer, device='cuda'):
        """
        编码文本为稀疏向量
        """
        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # 前向传播
        with torch.no_grad():
            sparse_vectors = self.forward(input_ids, attention_mask)
        
        return sparse_vectors


class SPLADETrainer:
    """
    SPLADE训练器
    """
    def __init__(self, model, tokenizer, device='cuda', flops_weight=0.0001):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.flops_weight = flops_weight
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-5,
            weight_decay=0.01
        )
    
    def compute_similarity(self, q_vectors, d_vectors):
        """
        计算查询和文档的相似度分数
        使用点积
        """
        # q_vectors: [batch, vocab_size]
        # d_vectors: [batch, vocab_size]
        scores = torch.sum(q_vectors * d_vectors, dim=-1)
        return scores
    
    def ranking_loss(self, q_vectors, pos_vectors, neg_vectors):
        """
        排序损失：确保正样本分数 > 负样本分数
        """
        pos_scores = self.compute_similarity(q_vectors, pos_vectors)
        neg_scores = self.compute_similarity(q_vectors, neg_vectors)
        
        # Margin ranking loss
        loss = -torch.mean(
            torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        )
        return loss
    
    def flops_regularization(self, vectors):
        """
        FLOPS正则化：鼓励稀疏性
        L1范数惩罚
        """
        return torch.mean(torch.sum(torch.abs(vectors), dim=-1))
    
    def train_step(self, queries, pos_docs, neg_docs):
        """
        单步训练
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 编码查询
        q_encoded = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # 编码正样本文档
        pos_encoded = self.tokenizer(
            pos_docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # 编码负样本文档
        neg_encoded = self.tokenizer(
            neg_docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # 前向传播
        q_vectors = self.model(q_encoded['input_ids'], q_encoded['attention_mask'])
        pos_vectors = self.model(pos_encoded['input_ids'], pos_encoded['attention_mask'])
        neg_vectors = self.model(neg_encoded['input_ids'], neg_encoded['attention_mask'])
        
        # 计算损失
        rank_loss = self.ranking_loss(q_vectors, pos_vectors, neg_vectors)
        
        # FLOPS正则化
        flops_loss = (
            self.flops_regularization(q_vectors) +
            self.flops_regularization(pos_vectors) +
            self.flops_regularization(neg_vectors)
        ) / 3.0
        
        # 总损失
        total_loss = rank_loss + self.flops_weight * flops_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'rank_loss': rank_loss.item(),
            'flops_loss': flops_loss.item()
        }


class SPLADERetriever:
    """
    SPLADE检索器：使用稀疏向量进行高效检索
    """
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # 文档索引：倒排索引结构
        self.doc_vectors = []  # 存储文档向量
        self.doc_ids = []  # 文档ID
        self.inverted_index = {}  # term_id -> [(doc_idx, weight)]
    
    def add_documents(self, documents: List[str], doc_ids: List[str] = None):
        """
        添加文档到索引
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.model.eval()
        
        # 批量编码文档
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            
            # 编码
            vectors = self.model.encode(batch_docs, self.tokenizer, self.device)
            vectors = vectors.cpu().numpy()
            
            # 构建倒排索引
            for doc_idx, (vector, doc_id) in enumerate(zip(vectors, batch_ids)):
                global_doc_idx = len(self.doc_vectors)
                self.doc_vectors.append(vector)
                self.doc_ids.append(doc_id)
                
                # 找到非零项
                nonzero_indices = np.nonzero(vector)[0]
                for term_id in nonzero_indices:
                    weight = vector[term_id]
                    if term_id not in self.inverted_index:
                        self.inverted_index[term_id] = []
                    self.inverted_index[term_id].append((global_doc_idx, weight))
        
        print(f"Indexed {len(self.doc_vectors)} documents")
    
    def search(self, query: str, top_k: int = 10):
        """
        搜索查询
        """
        self.model.eval()
        
        # 编码查询
        q_vector = self.model.encode([query], self.tokenizer, self.device)
        q_vector = q_vector.cpu().numpy()[0]
        
        # 找到查询中的非零项
        q_nonzero = np.nonzero(q_vector)[0]
        
        # 使用倒排索引找候选文档
        candidate_scores = {}
        for term_id in q_nonzero:
            if term_id in self.inverted_index:
                q_weight = q_vector[term_id]
                for doc_idx, d_weight in self.inverted_index[term_id]:
                    if doc_idx not in candidate_scores:
                        candidate_scores[doc_idx] = 0.0
                    # 点积累加
                    candidate_scores[doc_idx] += q_weight * d_weight
        
        # 排序
        sorted_docs = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 返回结果
        results = []
        for doc_idx, score in sorted_docs:
            results.append({
                'doc_id': self.doc_ids[doc_idx],
                'score': score,
                'doc_idx': doc_idx
            })
        
        return results
    
    def explain_query(self, query: str, top_terms: int = 10):
        """
        解释查询：显示激活的词汇
        """
        self.model.eval()
        
        # 编码查询
        q_vector = self.model.encode([query], self.tokenizer, self.device)
        q_vector = q_vector.cpu().numpy()[0]
        
        # 找到top激活的词
        top_indices = np.argsort(q_vector)[-top_terms:][::-1]
        
        print(f"\nQuery: '{query}'")
        print(f"Top {top_terms} activated terms:")
        print("-" * 50)
        
        for idx in top_indices:
            if q_vector[idx] > 0:
                term = self.tokenizer.decode([idx])
                weight = q_vector[idx]
                print(f"  {term:20s} : {weight:.4f}")
        
        # 统计稀疏度
        nonzero_count = np.count_nonzero(q_vector)
        sparsity = 1 - (nonzero_count / len(q_vector))
        print(f"\nSparsity: {sparsity:.4f} ({nonzero_count}/{len(q_vector)} non-zero)")


# ============= 使用示例 =============

def example_usage():
    """
    完整使用示例
    """
    print("=" * 60)
    print("SPLADE Model - Complete Example")
    print("=" * 60)
    
    # 1. 初始化模型
    print("\n1. Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = SPLADEModel('bert-base-uncased')
    model.to(device)
    print(f"Model loaded on {device}")
    
    # 2. 编码示例
    print("\n2. Encoding examples...")
    queries = [
        "machine learning tutorial",
        "python programming guide"
    ]
    
    model.eval()
    q_vectors = model.encode(queries, tokenizer, device)
    print(f"Query vectors shape: {q_vectors.shape}")
    print(f"Sparsity: {(q_vectors == 0).float().mean().item():.4f}")
    
    # 3. 显示激活的词汇
    print("\n3. Analyzing activated terms...")
    for i, query in enumerate(queries):
        vector = q_vectors[i].cpu().numpy()
        top_10_indices = np.argsort(vector)[-10:][::-1]
        
        print(f"\nQuery: '{query}'")
        print("Top 10 activated terms:")
        for idx in top_10_indices:
            if vector[idx] > 0:
                term = tokenizer.decode([idx])
                print(f"  {term:15s} : {vector[idx]:.4f}")
    
    # 4. 检索示例
    print("\n4. Retrieval example...")
    retriever = SPLADERetriever(model, tokenizer, device)
    
    # 添加文档
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret visual information",
        "A tutorial on machine learning covers algorithms and applications",
        "Programming in Python requires understanding of syntax and libraries",
        "Reinforcement learning trains agents through rewards and penalties"
    ]
    
    print("Building document index...")
    retriever.add_documents(documents)
    
    # 搜索
    query = "machine learning tutorial"
    print(f"\nSearching for: '{query}'")
    results = retriever.search(query, top_k=5)
    
    print("\nTop 5 results:")
    for rank, result in enumerate(results, 1):
        doc_idx = result['doc_idx']
        print(f"{rank}. Score: {result['score']:.4f}")
        print(f"   {documents[doc_idx][:80]}...")
    
    # 5. 查询解释
    print("\n5. Query explanation...")
    retriever.explain_query(query, top_terms=15)
    
    # 6. 训练示例（简化版）
    print("\n6. Training example (simplified)...")
    trainer = SPLADETrainer(model, tokenizer, device, flops_weight=0.0001)
    
    # 模拟训练数据
    train_queries = ["machine learning"]
    train_pos_docs = ["Machine learning is a subset of AI"]
    train_neg_docs = ["Python is a programming language"]
    
    loss_info = trainer.train_step(train_queries, train_pos_docs, train_neg_docs)
    print(f"Training step completed:")
    print(f"  Total loss: {loss_info['total_loss']:.4f}")
    print(f"  Ranking loss: {loss_info['rank_loss']:.4f}")
    print(f"  FLOPS loss: {loss_info['flops_loss']:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行示例
    example_usage()