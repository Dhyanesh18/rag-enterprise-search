import numpy as np
import hashlib

class BanditReranker:
    def __init__(self, exploration_rate=0.2, decay_factor=0.95):
        self.arm_stats = {}  # {chunk_hash: (count, total_score)}
        self.explore = exploration_rate
        self.decay = decay_factor
    
    def rerank(self, query, passages):
        scored = []
        for p in passages:
            chunk_hash = self._get_hash(p["text"])
            count, total = self.arm_stats.get(chunk_hash, (1, 0.5))  # Default pseudo-count
            
            # Upper Confidence Bound (UCB) strategy
            if np.random.random() < self.explore:
                score = np.random.random()  # Exploration
            else:
                score = (total / count) + np.sqrt(2 * np.log(sum(self.arm_stats.values())) / count)
            
            scored.append((score, p))
        
        return [p for _, p in sorted(scored, reverse=True)]
    
    def update(self, chosen_chunk, reward):
        chunk_hash = self._get_hash(chosen_chunk["text"])
        count, total = self.arm_stats.get(chunk_hash, (0, 0))
        self.arm_stats[chunk_hash] = (
            count * self.decay + 1,
            total * self.decay + reward
        )
    
    def _get_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()