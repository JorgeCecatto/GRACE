"""
Meta-Prompting: Melhora prompt inicial sem ver exemplos.

Transforma prompt simples em prompt de especialista generalista
antes de iniciar a otimização GRACE.
"""

from typing import List
from .prompts.meta_prompts import expert_generalist, zs_prompt, task_decomposition, meta_task_description
class MetaPrompter:
    """
    Gera versões melhoradas do prompt inicial usando meta-prompting.
    """
    
    def __init__(self, optim_model, task_description, strategy, logger):
        """
        Args:
            optim_model: Modelo LLM com método batch_forward_func
            task_description: Descrição da tarefa
            strategy: Estratégia de meta-prompting (expert_generalist, task_decomposition, zero_shot_cot)
            logger: Logger para mensagens
        """
        self.optim_model = optim_model
        self.task_description = task_description
        self.strategy = strategy
        self.logger = logger

    def create_meta_task_description(self, init_prompt: str, example: List[str]) -> str:
        """
        Cria descrição de tarefa para meta-prompting.
        
        Args:
            init_prompt: Prompt inicial simples
            example: Exemplo de entrada para a tarefa"""
        return meta_task_description.format(task_description=self.task_description, examples=example)
    
    def _create_meta_prompt(self, init_prompt: str, example: List[str]) -> str:
        """
        Cria meta-prompt para reescrever o prompt inicial.
        """
        new_meta_task_description = self.create_meta_task_description(init_prompt, example)
        new_task_description = self.optim_model.batch_forward_func([new_meta_task_description])[0]
        self.logger.info(f"New task description for meta-prompting:\n{new_task_description}\n")
        if self.strategy == 'expert_generalist':
            return expert_generalist.format(init_prompt=init_prompt, task_description=new_task_description)
        
        elif self.strategy == 'task_decomposition':
            return task_decomposition.format(init_prompt=init_prompt, task_description=new_task_description)
        
        elif self.strategy == 'zero_shot_cot':
            return zs_prompt.format(init_prompt=init_prompt)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


    
    def generate_improved_prompts(
        self, 
        init_prompt: str, 
        task_description: str,
        strategy: str = None,
        num_variants: int = 3,
        examples: List[str] = None
    ) -> List[str]:
        """
        Gera múltiplas versões melhoradas do prompt inicial usando UMA estratégia.
        
        Args:
            init_prompt: Prompt inicial simples
            task_description: Descrição da task
            strategy: Estratégia a usar (expert_generalist, task_decomposition, zero_shot_cot)
            num_variants: Quantas variantes gerar com essa estratégia
            examples: Exemplos para entender a task
            
        Returns:
            Lista de prompts melhorados
        """
        if strategy is None:
            strategy = 'expert_generalist'
        
        # Define a estratégia para uso interno
        self.strategy = strategy
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎭 META-PROMPTING")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Initial prompt: {init_prompt}")
        self.logger.info(f"Strategy: {strategy}")
        self.logger.info(f"Generating {num_variants} variants...")
        
        improved_prompts = [init_prompt]
        
        for i in range(num_variants):
            try:
                self.logger.info(f"\n  Variant {i+1}/{num_variants}...")
                
                # Cria meta-prompt
                meta_prompt = self._create_meta_prompt(init_prompt, examples)
                
                # Gera prompt melhorado
                response = self.optim_model.batch_forward_func([meta_prompt])[0]
                
                # Limpa resposta
                improved = response.strip()
                
                # Remove aspas se tiver
                improved = improved.strip('"\'')
                
                # Valida
                if improved and len(improved) > 10 and improved != init_prompt:
                    improved_prompts.append(improved)
                    self.logger.info(f"  ✓ Generated: {improved[:100]}...")
                else:
                    self.logger.warning(f"  ✗ Invalid response, skipping")
            
            except Exception as e:
                self.logger.warning(f"  ✗ Failed: {e}")
        
        self.logger.info(f"\n Generated {len(improved_prompts)} prompt variants")
        self.logger.info(f"{'='*80}\n")
        
        return improved_prompts[:num_variants]
    
    def select_best_prompt(
        self, 
        prompts: List[str], 
        eval_function
    ) -> str:
        """
        Avalia prompts e seleciona o melhor.
        
        Args:
            prompts: Lista de prompts candidatos
            eval_function: Função que avalia um prompt e retorna accuracy
            
        Returns:
            Melhor prompt
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" EVALUATING META-PROMPTS")
        self.logger.info(f"{'='*80}")
        
        best_prompt = prompts[0]
        best_score = 0.0
        
        for i, prompt in enumerate(prompts):
            try:
                self.logger.info(f"\n  Variant {i+1}/{len(prompts)}:")
                self.logger.info(f"  Prompt: {prompt[:80]}...")
                
                # Avalia
                score = eval_function(prompt)
                
                self.logger.info(f"  Score: {score:.3f}")
                
                # Em caso de empate (>=), prefere o mais recente
                if score >= best_score:
                    best_score = score
                    best_prompt = prompt
                    self.logger.info(f"  ⭐ New best!")
            
            except Exception as e:
                self.logger.warning(f"  ✗ Evaluation failed: {e}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f" BEST PROMPT (score={best_score:.3f}):")
        self.logger.info(f"{best_prompt}")
        self.logger.info(f"{'='*80}\n")
        
        return best_prompt
