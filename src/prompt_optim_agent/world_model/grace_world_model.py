from .gradient_descent import *
from ..test_helper import eval_instruction_with_loader
from typing import Generic
from ..search_algo.base_algo import State, Action
from ..search_algo.grace_search import GraceNode
from ..curriculum.curriculum_classifier import CurriculumClassifier
from .prompt_cache import PromptEvaluationCache

from collections import deque
import random
import numpy as np

class GraceSearchWorldModel():
    def __init__(
        self,
        task,
        logger,
        
        # model
        base_model: str,
        optim_model: str,
        iteration_num = 80,
        stop_early_thresh = 5,
        num_wrong_sample = 3,
        num_correct_sample = 3,
        num_new_prompts = 1,
        train_shuffle = True,
        train_batch_size: int = 8,
        test_batch_size: int = 200,
        eval_batch_size: int = 200,
        **kwargs
        ) -> None:
        
        self.task = task
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model

        self.iteration_num = iteration_num
        self.stop_early_thresh = stop_early_thresh
        self.num_correct_sample = num_correct_sample
        self.num_wrong_sample = num_wrong_sample

        self.train_dataloader = self.task.get_dataloader('train', 
                                                        batch_size=train_batch_size, 
                                                        shuffle=train_shuffle)
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)
        self.buffer = deque()
        
        self.test_dataloader = self.task.get_dataloader('test', 
                                                        batch_size=test_batch_size, 
                                                        shuffle=False)
        self.eval_dataloader = self.task.get_dataloader('eval', 
                                                        batch_size=eval_batch_size, 
                                                        shuffle=False)
        self.gradient_descent = GradientDescent(task=self.task, 
                                                logger=self.logger, 
                                                base_model=base_model, 
                                                optim_model=optim_model, 
                                                num_new_prompts = num_new_prompts,
                                                eval_dataloader = self.eval_dataloader)
        
        # Initialize prompt evaluation cache
        self.prompt_cache = PromptEvaluationCache(logger=self.logger)
    def _infinite_data_loader(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch
                
    def get_train_batch(self):
        return next(self.train_data_iterator)
    


    

    def sample_forward_output(self, forward_output, num_wrong=3, num_right=3, momentum='EASY'):
        """
        Sample examples baseado no momentum do nó.
        
        Momentum determina a dificuldade dos exemplos selecionados:
        - EASY: Prioriza exemplos EASY e MEDIUM
        - MEDIUM: Balanceado entre todas as dificuldades
        - HARD: Prioriza exemplos MEDIUM e HARD
        """
        examples = forward_output['examples']

        if "ncbi" not in self.task.task_name.lower():
            valid_examples = [ex for ex in examples if "format error" not in ex['pred'].lower()]
        else:
            valid_examples = examples
    
        # Enriquecer exemplos com dificuldade do curriculum
        for ex in valid_examples:
            if hasattr(self, 'difficulty_map') and ex.get('question') in self.difficulty_map:
                ex['difficulty'] = self.difficulty_map[ex['question']]
            else:
                ex['difficulty'] = 'MEDIUM'  # Default
        
        # Separar em corretos e errados
        wrong_samples = [ex for ex in valid_examples if ex['label'] != ex['pred']]
        right_samples = [ex for ex in valid_examples if ex['label'] == ex['pred']]
        
        # Filtrar por momentum (curriculum-aware sampling)
        wrong_filtered = self._filter_by_momentum(wrong_samples, momentum)
        right_filtered = self._filter_by_momentum(right_samples, momentum)
        
        # Sample com fallback para lista original se filtro ficar vazio
        selected_wrong = random.sample(
            wrong_filtered if wrong_filtered else wrong_samples,
            min(num_wrong, len(wrong_filtered) if wrong_filtered else len(wrong_samples))
        )
        selected_right = random.sample(
            right_filtered if right_filtered else right_samples,
            min(num_right, len(right_filtered) if right_filtered else len(right_samples))
        )
        
        selected = selected_right + selected_wrong

        new_forward_output = {
            'cur_prompt': forward_output['cur_prompt'],
            'examples': selected,
            'correct': [int(ex['label'] == ex['pred']) for ex in selected],
            'acc': np.mean([ex['label'] == ex['pred'] for ex in selected])
        }
        return new_forward_output
    
    def _filter_by_momentum(self, examples, momentum):
        """
        Filtra exemplos baseado no momentum atual.
        
        Estratégia de Curriculum Learning:
        - EASY momentum → Foca em exemplos EASY e MEDIUM
        - MEDIUM momentum → Aceita todos os níveis
        - HARD momentum → Foca em exemplos MEDIUM e HARD
        """
        if momentum == 'EASY':
            return [ex for ex in examples if ex.get('difficulty', 'MEDIUM') in ['EASY', 'MEDIUM']]
        elif momentum == 'HARD':
            return [ex for ex in examples if ex.get('difficulty', 'MEDIUM') in ['MEDIUM', 'HARD']]
        else: 
            return examples
    


    def check_number(self,forward_output):
        examples = forward_output['examples']

        if "ncbi" not in self.task.task_name.lower():
            valid_examples = [ex for ex in examples if "format error" not in ex['pred'].lower()]
        else:
            valid_examples = examples
    
        wrong_samples = [ex for ex in valid_examples if ex['label'] != ex['pred']]
        right_samples = [ex for ex in valid_examples if ex['label'] == ex['pred']]
        if len(right_samples)>=self.num_correct_sample and len(wrong_samples)>=self.num_wrong_sample:
            return True
        return False
        
    def train_forward(self, cur_prompt, momentum='MEDIUM'):
        """
        Forward pass no conjunto de treino com exemplos ordenados por dificuldade.
        
        Anti-curriculum: Sempre exibe exemplos mais difíceis primeiro.
        
        Args:
            cur_prompt: Prompt atual
            momentum: Nível de momentum para filtrar exemplos
            
        Returns:
            Aggregated output com métricas de todos os batches processados
        """
        # Criar dataloader ordenado por dificuldade (HARD → MEDIUM → EASY)
        sorted_dataloader = self._create_sorted_dataloader(momentum)
        
        aggregated_output = {
            'cur_prompt': cur_prompt,
            'correct': [],
            'examples': [],
            'acc': []
        }
        
        for batch in sorted_dataloader:
            forward_output_cur = self.gradient_descent.forward(batch=batch, cur_prompt=cur_prompt)
            aggregated_output['correct'] += forward_output_cur['correct']
            aggregated_output['examples'] += forward_output_cur['examples']
            aggregated_output['acc'].append(forward_output_cur['acc'])
            if self.check_number(aggregated_output):
                break

        aggregated_output['acc'] = np.mean(aggregated_output['acc'])

        return aggregated_output
    
    def _create_sorted_dataloader(self, momentum='MEDIUM'):
        """
        Cria dataloader com ordenação adaptativa baseada no momentum.
        
        Todos os exemplos são incluídos, apenas a ordem muda:
        - EASY momentum: MEDIUM → HARD → EASY (embaralhado em cada grupo)
        - MEDIUM momentum: HARD → MEDIUM → EASY (embaralhado em cada grupo)
        - HARD momentum: HARD → MEDIUM → EASY (embaralhado em cada grupo)
        
        Args:
            momentum: Nível atual ('EASY', 'MEDIUM', 'HARD')
            
        Returns:
            DataLoader com exemplos ordenados e embaralhados por grupo
        """
        # Basicamente to tentando aplicar conceitos daqui https://arxiv.org/pdf/2206.14486 juntando com curriculum 
        all_examples = []
        for batch in self.train_dataloader:
            for q, a in zip(batch['question'], batch['answer']):
                difficulty = self.difficulty_map.get(q, 'MEDIUM')
                all_examples.append({
                    'question': q,
                    'answer': a,
                    'difficulty': difficulty
                })
        
        # Separar por dificuldade
        hard_examples = [ex for ex in all_examples if ex['difficulty'] == 'HARD']
        medium_examples = [ex for ex in all_examples if ex['difficulty'] == 'MEDIUM']
        easy_examples = [ex for ex in all_examples if ex['difficulty'] == 'EASY']
        
        # Embaralhar cada grupo individualmente
        random.shuffle(hard_examples)
        random.shuffle(medium_examples)
        random.shuffle(easy_examples)
        
        # Ordenar baseado no momentum
        if momentum == 'EASY':
            # Começa do médio → hard → easy
            sorted_examples = medium_examples + hard_examples + easy_examples
            order_desc = "MEDIUM → HARD → EASY"
        elif momentum == 'HARD':
            # Começa do hard → medium → easy
            sorted_examples = hard_examples + medium_examples + easy_examples
            order_desc = "HARD → MEDIUM → EASY"
        else:  # MEDIUM
            # Começa do hard → medium → easy
            #tentei usar o conteido desse https://arxiv.org/pdf/2206.14486, sem mudar mt minha implementação
            sorted_examples = hard_examples + medium_examples + easy_examples
            order_desc = "HARD → MEDIUM → EASY"
        
        # Log da distribuição
        self.logger.info(
            f" Dataloader ordenado (momentum={momentum}): {order_desc}\n"
            f"   HARD={len(hard_examples)}, MEDIUM={len(medium_examples)}, EASY={len(easy_examples)}\n"
            f"   Total: {len(sorted_examples)} exemplos (embaralhados dentro de cada grupo)"
        )
        
        # Criar batches mantendo a ordem
        from torch.utils.data import Dataset, DataLoader
        
        class SortedDataset(Dataset):
            def __init__(self, examples):
                self.examples = examples
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                return self.examples[idx]
        
        dataset = SortedDataset(sorted_examples)
        
        # Criar dataloader sem shuffle para manter ordem
        sorted_loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,  # Mantém ordem definida acima
            collate_fn=lambda batch: {
                'question': [ex['question'] for ex in batch],
                'answer': [ex['answer'] for ex in batch]
            }
        )
        
        return sorted_loader
        

    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
        
    def _gradient_descent_step(self, node: GraceNode):

        new_nodes = []
        child_node = node
       
        # Herdar cache do nó pai para comparação de prompts
        if node.parent is not None:
            self.logger.info(f"Herdando cache do nó pai {node.parent.id}")

        #Get eval batch
        eval_batch = {"question":[],'answer':[]}
        for batch in self.eval_dataloader:
            eval_batch['question']+=batch['question']
            eval_batch['answer']+=batch['answer']


        #Get initial eval score
        eval_forward_output = self.gradient_descent.forward(batch=eval_batch, cur_prompt=child_node.prompt)
        child_node.eval_metric = eval_forward_output['acc']
        max_acc = self._sort_helper(eval_forward_output['acc'])

        #Split train into correct and wrong samples
        
        stop_early = 0

        for iter in range(self.iteration_num):
            cur_child_node = child_node
            self.logger.info(f'------------------  expand node {cur_child_node.id} ---------------------')

            # Pass momentum to train_forward
            train_forward_output = self.train_forward(
                cur_prompt=child_node.prompt,
                momentum=child_node.momentum if hasattr(child_node, 'momentum') else 'MEDIUM'
            )
            cur_acc = self._sort_helper(train_forward_output['acc'])
            
            # Atualizar train_accuracy e momentum do nó baseado na performance
            child_node.train_accuracy = cur_acc
            child_node.momentum = self._calculate_momentum(cur_acc)
            
            self.logger.info(f' Node {cur_child_node.id} - Train Accuracy: {cur_acc:.3f} | Momentum: {child_node.momentum}')
            
            if int(cur_acc)==1:
                break
           
            sampled_forward_output = self.sample_forward_output(
                train_forward_output,
                self.num_wrong_sample,
                self.num_correct_sample,
                momentum=child_node.momentum
            )


            self.logger.info(f'----------------  OPTIMIZATION batch {iter} ----------------')
            optimized_prompts = self.gradient_descent.step_wrong(child_node.prompt, forward_output = sampled_forward_output)
            for opt_prompt in optimized_prompts:
                # Tentar buscar no cache primeiro
                cached_acc = self.prompt_cache.get(
                    prompt=opt_prompt,
                    momentum=child_node.momentum,
                    current_acc=child_node.train_accuracy
                )
                
                if cached_acc is not None:
                    # Cache hit - usar acurácia cached
                    eval_temp_acc = cached_acc
                else:
                    # Cache miss - avaliar e adicionar ao cache
                    eval_temp_forward_output = self.gradient_descent.forward(
                        batch=eval_batch, 
                        cur_prompt=opt_prompt
                    )
                    eval_temp_acc = eval_temp_forward_output['acc']
                    
                    # Armazenar no cache
                    self.prompt_cache.put(
                        prompt=opt_prompt,
                        accuracy=eval_temp_acc,
                        momentum=child_node.momentum
                    )
                if iter % 10 == 0:
                    stats = self.prompt_cache.get_stats()
                    self.logger.info(f"Estatísticas do Cache: Taxa de Acertos={stats['hit_rate']:.1f}%, "
                                   f"Hits={stats['exact_hits'] + stats['similarity_hits']}, "
                                   f"Misses={stats['misses']}")
                
                temp_child_node = GraceNode(
                    prompt=opt_prompt, 
                    action="OPT",
                    mom_prompt=None,
                    parent=cur_child_node,
                    )
                temp_child_node.eval_metric = (eval_forward_output['acc'], eval_temp_acc)
                new_nodes.append(temp_child_node)
                
                #Change current prompt
                if self._sort_helper(eval_temp_acc)>max_acc:
                    stop_early = 0
                    max_acc = self._sort_helper(eval_temp_acc)
                    child_node = temp_child_node
                    #Re-Split train into correct and wrong samples
                    eval_forward_output = {'acc': eval_temp_acc}
                else:
                    stop_early+=1

            if stop_early==self.stop_early_thresh:
                self.logger.info(f'----------------  SIMPLIFY batch {iter} ----------------')
                simp_prompt = self.gradient_descent.step_simp(child_node.prompt)
                
                # Tentar buscar no cache primeiro
                cached_acc = self.prompt_cache.get(
                    prompt=simp_prompt,
                    momentum=child_node.momentum,
                    current_acc=child_node.train_accuracy
                )
                
                if cached_acc is not None:
                    # Cache hit - usar acurácia cached
                    eval_temp_acc = cached_acc
                else:
                    # Cache miss - avaliar e adicionar ao cache
                    eval_temp_forward_output = self.gradient_descent.forward(
                        batch=eval_batch, 
                        cur_prompt=simp_prompt
                    )
                    eval_temp_acc = eval_temp_forward_output['acc']
                    
                    # Armazenar no cache
                    self.prompt_cache.put(
                        prompt=simp_prompt,
                        accuracy=eval_temp_acc,
                        momentum=child_node.momentum
                    )
                
                temp_child_node = GraceNode(
                    prompt=simp_prompt, 
                    action="SIMP",
                    mom_prompt=None,
                    parent=cur_child_node,
                    )
                temp_child_node.eval_metric = (eval_forward_output['acc'], eval_temp_acc)
                new_nodes.append(temp_child_node)
                
                stop_early = 0
                max_acc = self._sort_helper(eval_temp_acc)
                child_node = temp_child_node
                eval_forward_output = {'acc': eval_temp_acc}
        
        # Log final cache statistics
        stats = self.prompt_cache.get_stats()
        total_hits = stats['exact_hits'] + stats['similarity_hits']
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Final Cache Statistics for Node {node.id}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Hit Rate: {stats['hit_rate']:.1%}")
        self.logger.info(f"Exact Hits: {stats['exact_hits']}")
        self.logger.info(f"Similarity Hits: {stats['similarity_hits']}")
        self.logger.info(f"Total Hits: {total_hits}")
        self.logger.info(f"Cache Misses: {stats['misses']}")
        self.logger.info(f"Total Queries: {stats['total_queries']}")
        self.logger.info(f"Time Saved (estimated): ~{total_hits * 90}s")
        self.logger.info(f"{'='*60}\n")

        return new_nodes, None


    
    def step(self, node:GraceNode):
        new_nodes, gradient_descent_output = self._gradient_descent_step(node=node)
        return new_nodes, gradient_descent_output
    
    def build_root(self, init_prompt):
        # Calculate curriculum for the training set
        self._calculate_curriculum(init_prompt)
        
        # Criar nó raiz
        node = GraceNode(prompt=init_prompt, action=None, parent=None, mom_prompt=None)
                
        # Avaliar acurácia do prompt inicial
        eval_batch = {"question": [], "answer": []}
        for batch in self.eval_dataloader:
            eval_batch['question'] += batch['question']
            eval_batch['answer'] += batch['answer']
        
        initial_eval = self.gradient_descent.forward(batch=eval_batch, cur_prompt=init_prompt)
        initial_acc = initial_eval['acc']
        
        # Adicionar prompt inicial ao cache com momentum MEDIUM (neutro), é importante pq as vzs saem prompt identicos a raíz.
        self.prompt_cache.put(
            prompt=init_prompt,
            accuracy=initial_acc,
            momentum='MEDIUM'
        )
        
        node.eval_metric = initial_acc
        
        self.logger.info(f"Root node added to cache:")

        return node
    
    def _calculate_curriculum(self, init_prompt):
        """
        Calculate curriculum difficulty for all training examples.
        """
        self.logger.info("\n" + "------"*40)
        self.logger.info("CALCULATING CURRICULUM FOR TRAINING SET")
        self.logger.info("-----"*40)
        
        examples = []
        for batch in self.train_dataloader:
            for q, a in zip(batch['question'], batch['answer']):
                examples.append({'question': q, 'answer': a})
        
        self.logger.info(f"Processing {len(examples)} training examples...")
        
        # Initialize classifier
        classifier = CurriculumClassifier(
            task=self.task,
            base_model=self.base_model,
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            logger=self.logger
        )
        
        # Classify dataset
        self.curriculum_examples = classifier.classify_dataset(examples, init_prompt)
        
        # Create difficulty map for quick lookup
        self.difficulty_map = {ex.question: ex.difficulty for ex in self.curriculum_examples}
        self.logger.info("✅ Curriculum calculation complete.")
    
    def _calculate_momentum(self, train_accuracy):
        """
        Calcula momentum baseado na acurácia no treino.
        
        Estratégia Adaptativa de Curriculum Learning:
        - Alta acurácia (>= 0.7) → HARD: Modelo está dominando, aumentar dificuldade
        - Média acurácia (0.4-0.7) → MEDIUM: Balancear dificuldade
        - Baixa acurácia (< 0.4) → EASY: Modelo está lutando, focar em exemplos mais fáceis
        
        Args:
            train_accuracy: Acurácia atual no conjunto de treino (0.0 a 1.0)
            
        Returns:
            momentum: 'EASY', 'MEDIUM' ou 'HARD'
        """
        if train_accuracy >= 0.7:
            return 'HARD'
        elif train_accuracy >= 0.4:
            return 'MEDIUM'
        else:
            return 'EASY'

    def test_prompt(self, prompt):
        metric, eval_output = eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.test_dataloader,
                                           base_model=self.base_model,
                                           )
        return metric, eval_output
    


