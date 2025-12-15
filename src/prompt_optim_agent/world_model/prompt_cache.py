"""
Sistema de cache de avaliação de prompts baseado em similaridade semântica.

Economiza tempo pulando avaliações de prompts muito similares,
com thresholds adaptativos baseados no momentum e acurácia.

ABORDAGEM SEMÂNTICA:
- Usa embeddings (SentenceTransformers) para capturar significado
- Calcula cosine similarity entre embeddings de prompts
- Distância semântica = 1 - cosine_similarity
- Mais robusto que substring matching (difflib)

EXEMPLO DE USO:
    parent_prompt = "Classify the sentiment of the following text"
    candidate_prompt = "Determine if the text below is positive or negative"
    
    # Gerar embeddings
    parent_embedding = model.encode(parent_prompt)
    candidate_embedding = model.encode(candidate_prompt)
    
    # Calcular similaridade semântica
    cos_sim = util.pytorch_cos_sim(candidate_embedding, parent_embedding).item()
    # cos_sim ≈ 0.85 (alto, pois prompts têm mesmo significado)
    
    # Distância semântica
    dist_to_parent = 1 - cos_sim  # ≈ 0.15 (baixo)
    
    # Decisão: Se cos_sim >= threshold → usar cache
    if cos_sim >= 0.85:  # threshold para HARD+low_acc
        return cached_accuracy
"""

import hashlib
import numpy as np
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, util
import torch


class PromptEvaluationCache:
    """
    Cache inteligente para avaliações de prompts.
    
    Features:
    - Comparação exata por hash
    - Comparação semântica via embeddings (cosine similarity)
    - Thresholds adaptativos por momentum e acurácia
    - Limite de tamanho (LRU)
    """
    
    def __init__(
        self,
        logger,
        enable_cache: bool = True,
        max_cache_size: int = 100,
        high_acc_threshold: float = 0.70,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ):
        """
        Args:
            logger: Logger para output
            enable_cache: Habilita/desabilita cache
            max_cache_size: Número máximo de entradas no cache
            high_acc_threshold: Threshold para considerar "alta acurácia"
            embedding_model: Nome do modelo de embeddings para similaridade semântica
        """
        self.logger = logger
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.high_acc_threshold = high_acc_threshold
        
        # Inicializar modelo de embeddings
        self.logger.info(f"🔧 Carregando modelo de embeddings: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.logger.info("✅ Modelo de embeddings carregado com sucesso")
        
        # Cache: {prompt_hash: {
        #   'prompt': str, 
        #   'accuracy': float, 
        #   'momentum': str,
        #   'embedding': tensor
        # }}
        self.cache = OrderedDict()
        
        # Thresholds de similaridade adaptativos (baseado em cosine similarity)
        self.similarity_thresholds = {
            'HARD': {
                'high_acc': 0.98,   # Rigoroso: pequenas mudanças importam
                'low_acc': 0.85     # Relaxado: precisa mudanças grandes
            },
            'MEDIUM': 0.95,
            'EASY': 0.93
        }
        
        # Estatísticas
        self.stats = {
            'exact_hits': 0,
            'similarity_hits': 0,
            'misses': 0,
            'evaluations_saved': 0
        }
    
    def _hash_prompt(self, prompt: str) -> str:
        """Gera hash MD5 do prompt."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def _get_embedding(self, prompt: str) -> torch.Tensor:
        """
        Gera embedding semântico do prompt.
        
        Args:
            prompt: Texto do prompt
            
        Returns:
            Tensor com embedding do prompt
        """
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                prompt, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
        return embedding
    
    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Calcula similaridade semântica entre dois prompts usando cosine similarity.
        
        Abordagem semântica (embeddings) ao invés de substring matching.
        Isso captura significado, não apenas caracteres.
        
        Args:
            prompt1: Primeiro prompt
            prompt2: Segundo prompt
            
        Returns:
            Cosine similarity entre 0.0 e 1.0 (1.0 = semanticamente idênticos)
        """
        if prompt1 == prompt2:
            return 1.0
        
        # Gerar embeddings
        embedding1 = self._get_embedding(prompt1)
        embedding2 = self._get_embedding(prompt2)
        
        # Calcular cosine similarity
        cos_sim = util.pytorch_cos_sim(embedding1, embedding2).item()
        
        # Converter para similaridade (0-1, onde 1 = idêntico)
        # Cosine similarity já retorna valores entre -1 e 1, 
        # mas para embeddings de texto geralmente fica entre 0 e 1
        similarity = max(0.0, cos_sim)  # Garante não-negativo
        
        return similarity
    
    def _calculate_semantic_distance(self, prompt1: str, prompt2: str) -> float:
        """
        Calcula distância semântica entre dois prompts.
        
        Distância = 1 - similaridade
        
        Args:
            prompt1: Primeiro prompt
            prompt2: Segundo prompt
            
        Returns:
            Distância semântica entre 0.0 e 1.0 (0.0 = idênticos)
        """
        similarity = self._calculate_similarity(prompt1, prompt2)
        distance = 1.0 - similarity
        return distance
    
    def _get_similarity_threshold(self, momentum: str, current_acc: float) -> float:
        """
        Retorna threshold de similaridade adaptativo.
        
        Estratégia:
        - HARD + Alta acurácia (>70%): Threshold alto (0.98)
          → Pequenas mudanças podem fazer diferença em exemplos difíceis
        - HARD + Baixa acurácia (<70%): Threshold baixo (0.85)
          → Precisa de mudanças significativas para sair do mínimo local
        - MEDIUM/EASY: Thresholds fixos
        
        Args:
            momentum: Nível de dificuldade atual ('EASY', 'MEDIUM', 'HARD')
            current_acc: Acurácia atual (0.0 a 1.0)
            
        Returns:
            Threshold de similaridade (0.0 a 1.0)
        """
        if momentum == 'HARD':
            if current_acc > self.high_acc_threshold:
                threshold = self.similarity_thresholds['HARD']['high_acc']
                self.logger.debug(
                    f"HARD + High Acc ({current_acc:.1%}) → "
                    f"Strict threshold: {threshold}"
                )
            else:
                threshold = self.similarity_thresholds['HARD']['low_acc']
                self.logger.debug(
                    f"HARD + Low Acc ({current_acc:.1%}) → "
                    f"Relaxed threshold: {threshold}"
                )
        else:
            threshold = self.similarity_thresholds[momentum]
            self.logger.debug(f"{momentum} → Threshold: {threshold}")
        
        return threshold
    
    def get(
        self, 
        prompt: str, 
        momentum: str = 'MEDIUM',
        current_acc: float = 0.5
    ) -> Optional[float]:
        """
        Busca avaliação no cache comparando com TODOS os prompts armazenados.
        
        Usa busca vetorizada para encontrar o prompt mais similar em todo o cache.
        
        Args:
            prompt: Prompt a buscar
            momentum: Momentum atual para threshold adaptativo
            current_acc: Acurácia atual para threshold adaptativo
            
        Returns:
            Acurácia do cache se encontrado, None caso contrário
        """
        if not self.enable_cache:
            return None
        
        prompt_hash = self._hash_prompt(prompt)
        
        # 1. Tenta match exato
        if prompt_hash in self.cache:
            cached_data = self.cache[prompt_hash]
            self.stats['exact_hits'] += 1
            self.logger.info(
                f" Cache EXACT HIT! "
                f"Acc: {cached_data['accuracy']:.2%}"
            )
            # Move para o final (LRU)
            self.cache.move_to_end(prompt_hash)
            return cached_data['accuracy']
        
        # 2. Busca por similaridade semântica em TODOS os prompts do cache
        if len(self.cache) > 0:
            threshold = self._get_similarity_threshold(momentum, current_acc)
            
            # Gerar embedding do prompt atual UMA VEZ
            current_embedding = self._get_embedding(prompt)
            
            # Coletar todos os embeddings e dados do cache
            cache_embeddings = []
            cache_data_list = []
            cache_hashes = []
            
            for cache_hash, cached_data in self.cache.items():
                cache_embeddings.append(cached_data['embedding'])
                cache_data_list.append(cached_data)
                cache_hashes.append(cache_hash)
            
            # Stack embeddings em um tensor 2D para busca vetorizada
            cache_embeddings_tensor = torch.stack(cache_embeddings)
            
            # Calcula similaridade com TODOS os prompts de uma vez (vetorizado)
            similarities = util.pytorch_cos_sim(current_embedding, cache_embeddings_tensor)[0]
            
            # Encontra o índice com maior similaridade
            best_idx = similarities.argmax().item()
            best_similarity = similarities[best_idx].item()
            best_match = cache_data_list[best_idx]
            best_hash = cache_hashes[best_idx]
            
            dist_to_best = 1 - best_similarity
            
            self.logger.debug(
                f" Compared with {len(self.cache)} prompts in cache "
                f"(vectorized search)"
            )
            
            # Verifica se o melhor match passa do threshold
            if best_similarity >= threshold:
                self.stats['similarity_hits'] += 1
                self.stats['evaluations_saved'] += 1
                
                self.logger.info(
                    f"   Cache SEMANTIC HIT! Found best match:\n"
                    f"   Cosine Sim: {best_similarity:.4f} >= {threshold:.3f}\n"
                    f"   Distance: {dist_to_best:.4f}\n"
                    f"   Matched prompt: '{best_match['prompt'][:70]}...'\n"
                    f"   Matched momentum: {best_match['momentum']}\n"
                    f"   Reusing acc: {best_match['accuracy']:.2%}"
                )
                
                # Adiciona entrada com hash do novo prompt apontando para mesma acc
                self.put(prompt, best_match['accuracy'], momentum, current_embedding)
                
                # Move o match usado para o final (mais recente no LRU)
                self.cache.move_to_end(best_hash)
                
                return best_match['accuracy']
            else:
                self.logger.info(
                    f"Cache MISS. Best match not similar enough:\n"
                    f"   Best Cosine Sim: {best_similarity:.4f} < {threshold:.3f}\n"
                    f"   Distance: {dist_to_best:.4f}"
                )
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    def put(
        self, 
        prompt: str, 
        accuracy: float, 
        momentum: str = 'MEDIUM',
        embedding: Optional[torch.Tensor] = None
    ):
        """
        Adiciona avaliação ao cache com seu embedding semântico.
        
        Args:
            prompt: Prompt avaliado
            accuracy: Acurácia obtida
            momentum: Momentum usado na avaliação
            embedding: Embedding pré-calculado (opcional, será gerado se None)
        """
        if not self.enable_cache:
            return
        
        prompt_hash = self._hash_prompt(prompt)
        
        # Gerar embedding se não fornecido
        if embedding is None:
            embedding = self._get_embedding(prompt)
        
        # Adiciona ao cache
        self.cache[prompt_hash] = {
            'prompt': prompt,
            'accuracy': accuracy,
            'momentum': momentum,
            'embedding': embedding
        }
        
        # Move para o final (mais recente)
        self.cache.move_to_end(prompt_hash)
        
        # Remove entradas antigas se exceder tamanho
        while len(self.cache) > self.max_cache_size:
            oldest_hash = next(iter(self.cache))
            removed = self.cache.pop(oldest_hash)
            self.logger.debug(
                f"🗑️ Cache cheio - removida a entrada mais antiga "
                f"(acc: {removed['accuracy']:.2%})"
            )
        
        self.logger.info(
            f"Cached evaluation - Acurácia Registrada: {accuracy:.2%} "
            f"(cache size: {len(self.cache)}/{self.max_cache_size})"
        )
    
    def clear(self):
        """Limpa o cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do cache."""
        total_queries = sum([
            self.stats['exact_hits'],
            self.stats['similarity_hits'],
            self.stats['misses']
        ])
        
        hit_rate = 0.0
        if total_queries > 0:
            hit_rate = (self.stats['exact_hits'] + self.stats['similarity_hits']) / total_queries
        
        return {
            **self.stats,
            'total_queries': total_queries,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
    
    def log_stats(self):
        """Loga estatísticas do cache."""
        stats = self.get_stats()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("CACHE STATISTICS")
        self.logger.info("="*60)
        self.logger.info(f"Total queries:        {stats['total_queries']}")
        self.logger.info(f"Exact hits:           {stats['exact_hits']}")
        self.logger.info(f"Similarity hits:      {stats['similarity_hits']}")
        self.logger.info(f"Misses:               {stats['misses']}")
        self.logger.info(f"Hit rate:             {stats['hit_rate']:.1%}")
        self.logger.info(f"Evaluations saved:    {stats['evaluations_saved']}")
        self.logger.info(f"Cache size:           {stats['cache_size']}/{self.max_cache_size}")
        self.logger.info("="*60 + "\n")
