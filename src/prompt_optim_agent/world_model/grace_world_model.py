from .gradient_descent import *
from ..test_helper import eval_instruction_with_loader
from typing import Generic
from ..search_algo.base_algo import State, Action
from ..search_algo.grace_search import GraceNode
from ..curriculum.curriculum_classifier import CurriculumClassifier
from ..curriculum.difficulty_tracker import DifficultyTracker, trace

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
        eval_temperature = None,

        # curriculum
        curriculum_mode: str = 'static',      # 'static' | 'bayes' | 'off'
        centroid_mode: str = 'global',        # 'class' | 'global'
        intrinsic_mode: str = 'centroid',     # 'centroid' | 'hdbscan'
        curriculum_window: int = 10,
        curriculum_prior_strength: float = 2.0,
        curriculum_refit_every: int = 5,
        curriculum_seed: int = 0,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: int = 3,
        typical_percentile: float = 50.0,
        min_group_size: int = 10,
        regression_threshold: float = 2.0,
        regression_min_T: int = 3,
        regression_easy_threshold: float = 0.80,
        **kwargs
        ) -> None:

        assert curriculum_mode in ('static', 'bayes', 'off'), f"curriculum_mode inválido: {curriculum_mode}"
        assert centroid_mode in ('class', 'global'), f"centroid_mode inválido: {centroid_mode}"
        assert intrinsic_mode in ('centroid', 'hdbscan'), f"intrinsic_mode inválido: {intrinsic_mode}"
        self.curriculum_mode = curriculum_mode
        self.centroid_mode = centroid_mode
        self.intrinsic_mode = intrinsic_mode
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.typical_percentile = typical_percentile
        self.min_group_size = min_group_size
        self.regression_threshold = regression_threshold
        self.regression_min_T = regression_min_T
        # Limiar de p_hat para entrar em T no teste de regressão ACUMULADO.
        # Separado do easy_threshold (0.85) que define os tiers EASY/MIXED/HARD.
        self.regression_easy_threshold = regression_easy_threshold
        self.curriculum_rng = np.random.default_rng(curriculum_seed)
        self.tracker = None
        if curriculum_mode == 'bayes':
            self.tracker = DifficultyTracker(
                window=curriculum_window,
                prior_strength=curriculum_prior_strength,
                refit_every=curriculum_refit_every,
                logger=logger,
            )
        self._last_sampling = {'regime_triggered': False, 'n_regressed_in_batch': 0}
        # Acumulação do teste de regressão por incumbente (ver _set_incumbent)
        self._inc_id = None
        self._inc_since_iter = 0
        self._inc_snapshot = {}
        self._inc_obs = {}  # key -> (correct, p_hat do snapshot); só a 1ª observação

        self.task = task
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model

        self.iteration_num = iteration_num
        self.stop_early_thresh = stop_early_thresh
        self.num_correct_sample = num_correct_sample
        self.num_wrong_sample = num_wrong_sample
        # Temperatura usada apenas nas chamadas de PONTUACAO (eval/test). None =
        # usa a temperatura configurada do modelo (comportamento historico);
        # 0.0 = avaliacao deterministica. Ver docs/curriculo_dois_regimes.md, Sec. 9.
        self.eval_temperature = eval_temperature

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
        if self.curriculum_mode == 'off':
            momentum = 'MEDIUM'  # GRACE original: sem filtro por dificuldade

        examples = forward_output['examples']

        if "ncbi" not in self.task.task_name.lower():
            valid_examples = [ex for ex in examples if "format error" not in ex['pred'].lower()]
        else:
            valid_examples = examples

        if self.curriculum_mode == 'bayes':
            return self._sample_forward_output_bayes(forward_output, valid_examples, num_wrong, num_right)

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

    def _sample_forward_output_bayes(self, forward_output, valid_examples, num_wrong, num_right):
        """
        Seleção do lote de atualização por contradição à expectativa da linhagem
        (DifficultyTracker.sample_update_batch). Se o teste de regressão do
        train_forward deu r > regression_threshold, os erros vêm primeiro dos
        exemplos regredidos.
        """
        wrong_samples = [ex for ex in valid_examples if ex['label'] != ex['pred']]
        right_samples = [ex for ex in valid_examples if ex['label'] == ex['pred']]

        regression = forward_output.get('regression') or {}
        r = regression.get('r')
        triggered = r is not None and r > self.regression_threshold
        regressed_keys = regression.get('regressed_keys', []) if triggered else None

        # p_hat anterior ao update desta iteração: mede a contradição com a
        # expectativa da linhagem sem a observação do prompt atual.
        p_hat_map = forward_output.get('p_hat_pre')
        wrong_idx, right_idx = self.tracker.sample_update_batch(
            [ex['question'] for ex in wrong_samples],
            [ex['question'] for ex in right_samples],
            num_wrong,
            num_right,
            self.curriculum_rng,
            regressed_keys=regressed_keys,
            p_hat_map=p_hat_map,
        )
        selected_wrong = [wrong_samples[i] for i in wrong_idx]
        selected_right = [right_samples[i] for i in right_idx]
        selected = selected_right + selected_wrong

        for ex in selected:
            key = ex['question']
            ex['tier'] = self.tracker.tier(key)
            p = p_hat_map.get(key) if p_hat_map else None
            ex['p_hat'] = p if p is not None else self.tracker.p_success(key)

        reg_set = set(regressed_keys or [])
        self._last_sampling = {
            'regime_triggered': triggered,
            'n_regressed_in_batch': sum(ex['question'] in reg_set for ex in selected_wrong),
        }

        return {
            'cur_prompt': forward_output['cur_prompt'],
            'examples': selected,
            'correct': [int(ex['label'] == ex['pred']) for ex in selected],
            'acc': np.mean([ex['label'] == ex['pred'] for ex in selected])
        }

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
        
    def train_forward(self,cur_prompt):
        aggregated_output = {
            'cur_prompt': cur_prompt,
            'correct': [],
            'examples': [],
            'acc': []
        }
        
        bayes = self.curriculum_mode == 'bayes'
        observations = []  # (key, correct, p_hat anterior ao update)
        p_hat_pre = {}

        for batch in self.train_dataloader:
            forward_output_cur = self.gradient_descent.forward(batch=batch, cur_prompt=cur_prompt)
            aggregated_output['correct'] += forward_output_cur['correct']
            aggregated_output['examples'] += forward_output_cur['examples']
            aggregated_output['acc'].append(forward_output_cur['acc'])
            if bayes:
                keys = [ex['question'] for ex in forward_output_cur['examples']]
                correct = [int(c) for c in forward_output_cur['correct']]
                # p_hat capturado ANTES do update (ver DifficultyTracker.regression_test)
                p_before = [self.tracker.p_success(k) for k in keys]
                self.tracker.update(keys, correct)
                for k, c, p in zip(keys, correct, p_before):
                    observations.append((k, c, p))
                    p_hat_pre[k] = p
            if self.check_number(aggregated_output):
                break

        aggregated_output['acc'] = np.mean(aggregated_output['acc'])

        if bayes:
            # r por iteração (comportamento anterior; só para log/comparação):
            # observações desta chamada, p_hat anterior ao update, limiar dos tiers.
            aggregated_output['regression_iter'] = self.tracker.regression_test(
                observations, min_T=self.regression_min_T
            )
            # r acumulado por incumbente (decide o regime de amostragem):
            # 1ª observação de cada exemplo com este incumbente, avaliada contra o
            # p_hat congelado na troca de incumbente. Repetições do mesmo exemplo
            # com o mesmo prompt são ignoradas (não entram em F, E, V nem |T|).
            if self._inc_id is None:
                self._set_incumbent(None, 0)
            for k, c, _ in observations:
                if k not in self._inc_obs:
                    self._inc_obs[k] = (c, self._inc_snapshot.get(k))
            aggregated_output['regression'] = self.tracker.regression_test(
                [(k, c, p) for k, (c, p) in self._inc_obs.items()],
                easy_threshold=self.regression_easy_threshold,
                min_T=self.regression_min_T,
            )
            aggregated_output['p_hat_pre'] = p_hat_pre

        return aggregated_output

    def _set_incumbent(self, node, iter):
        """
        Troca de incumbente (início do passo, OPT aceito ou SIMP): congela o
        p_hat da linhagem e zera as observações acumuladas.
        """
        self._inc_id = node.id if node is not None else -1
        self._inc_since_iter = iter
        self._inc_snapshot = self.tracker.snapshot_p_hat()
        self._inc_obs = {}
        trace(self.logger, 'incumbent', {'node_id': self._inc_id, 'since_iter': iter})
        

    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
        
    def _gradient_descent_step(self, node: GraceNode):

        new_nodes = []
        child_node = node
       
        #Get train batch
        # train_batch = {"question":[],'answer':[]}
        # for batch in self.train_dataloader:
        #     train_batch['question']+=batch['question']
        #     train_batch['answer']+=batch['answer']

        #Get eval batch
        eval_batch = {"question":[],'answer':[]}
        for batch in self.eval_dataloader:
            eval_batch['question']+=batch['question']
            eval_batch['answer']+=batch['answer']


        #Get initial eval score
        eval_forward_output = self.gradient_descent.forward(batch=eval_batch, cur_prompt=child_node.prompt, temperature=self.eval_temperature)
        child_node.eval_metric = eval_forward_output['acc']
        max_acc = self._sort_helper(eval_forward_output['acc'])
        if self.curriculum_mode == 'bayes':
            self._set_incumbent(child_node, 0)

        #Split train into correct and wrong samples

        stop_early = 0

        for iter in range(self.iteration_num):
            cur_child_node = child_node
            self.logger.info(f'------------------  expand node {cur_child_node.id} ---------------------')

            train_forward_output = self.train_forward(cur_prompt=child_node.prompt)
            cur_acc = self._sort_helper(train_forward_output['acc'])
            
            # Atualizar train_accuracy e momentum do nó baseado na performance
            child_node.train_accuracy = cur_acc
            if self.curriculum_mode == 'bayes':
                regression = train_forward_output['regression']            # acumulado
                regression_iter = train_forward_output['regression_iter']  # por iteração
                child_node.regression_r = regression['r']
                # Momentum só para log: no modo 'bayes' ele não filtra a amostragem.
                # TODO(etapa futura): competência c_t da linhagem com passos
                # assimétricos (subida lenta quando o candidato é aceito ou r é baixo,
                # queda rápida quando r > regression_threshold) entraria aqui,
                # substituindo este momentum derivado de max_acc. NÃO implementado.
                child_node.momentum = self._calculate_momentum(max_acc)
            else:
                child_node.momentum = self._calculate_momentum(cur_acc)

            self.logger.info(f' Node {cur_child_node.id} - Train Accuracy: {cur_acc:.3f} | Momentum: {child_node.momentum}')

            if int(cur_acc)==1:
                if self.curriculum_mode == 'bayes':
                    self._trace_iteration(iter, child_node, cur_acc, regression, regression_iter, stopped='train_acc_1')
                break

            sampled_forward_output = self.sample_forward_output(
                train_forward_output,
                self.num_wrong_sample,
                self.num_correct_sample,
                momentum=child_node.momentum
            )
            if self.curriculum_mode == 'bayes':
                self._trace_iteration(iter, child_node, cur_acc, regression, regression_iter)
                if self._last_sampling['regime_triggered']:
                    self._trace_trigger(iter, child_node, regression)


            self.logger.info(f'----------------  OPTIMIZATION batch {iter} ----------------')
            optimized_prompts = self.gradient_descent.step_wrong(child_node.prompt, forward_output = sampled_forward_output)
            for opt_prompt in optimized_prompts:
                eval_temp_forward_output = self.gradient_descent.forward(batch=eval_batch, cur_prompt=opt_prompt, temperature=self.eval_temperature)
                temp_child_node = GraceNode(
                    prompt=opt_prompt, 
                    action="OPT",
                    mom_prompt=None,
                    parent=cur_child_node,
                    )
                temp_child_node.eval_metric = (eval_forward_output['acc'],eval_temp_forward_output['acc'])
                new_nodes.append(temp_child_node)
                if self.curriculum_mode == 'bayes':
                    self._trace_candidate(iter, cur_child_node, temp_child_node, 'OPT',
                                          self._sort_helper(eval_temp_forward_output['acc']), max_acc,
                                          accepted=self._sort_helper(eval_temp_forward_output['acc'])>max_acc)

                #Change current prompt
                if self._sort_helper(eval_temp_forward_output['acc'])>max_acc:
                    stop_early = 0
                    max_acc = self._sort_helper(eval_temp_forward_output['acc'])
                    child_node = temp_child_node
                    if self.curriculum_mode == 'bayes':
                        self._set_incumbent(child_node, iter + 1)
                    #Re-Split train into correct and wrong samples
                    # train_forward_output = self.train_forward(cur_prompt=opt_prompt)
                    eval_forward_output = eval_temp_forward_output
                else:
                    stop_early+=1

            if stop_early==self.stop_early_thresh:
                self.logger.info(f'----------------  SIMPLIFY batch {iter} ----------------')
                simp_prompt = self.gradient_descent.step_simp(child_node.prompt)
                eval_temp_forward_output = self.gradient_descent.forward(batch=eval_batch, cur_prompt=simp_prompt, temperature=self.eval_temperature)
                temp_child_node = GraceNode(
                    prompt=simp_prompt, 
                    action="SIMP",
                    mom_prompt=None,
                    parent=cur_child_node,
                    )
                temp_child_node.eval_metric = (eval_forward_output['acc'],eval_temp_forward_output['acc'])
                new_nodes.append(temp_child_node)
                if self.curriculum_mode == 'bayes':
                    # SIMP é sempre aceito pelo GRACE (substitui o incumbente)
                    self._trace_candidate(iter, cur_child_node, temp_child_node, 'SIMP',
                                          self._sort_helper(eval_temp_forward_output['acc']), max_acc,
                                          accepted=True, source_node=child_node)

                stop_early = 0
                max_acc = self._sort_helper(eval_temp_forward_output['acc'])
                child_node = temp_child_node
                if self.curriculum_mode == 'bayes':
                    self._set_incumbent(child_node, iter + 1)
                # train_forward_output = self.train_forward(cur_prompt=opt_prompt)
                eval_forward_output = eval_temp_forward_output

        if self.curriculum_mode == 'bayes':
            self.tracker.log_state('step_end', with_items=True)

        return new_nodes, None

    def _trace_iteration(self, iter, node, train_acc, regression, reg_iter, stopped=None):
        # regression = acumulado por incumbente (decide o regime);
        # reg_iter = por iteração (comportamento anterior, só comparação).
        payload = {
            'iter': iter,
            'node_id': node.id,
            'train_acc': float(train_acc),
            'r_iter': reg_iter.get('r'),
            'F_iter': reg_iter.get('F'),
            'E_iter': reg_iter.get('E'),
            'T_iter': reg_iter.get('T'),
            'r_cum': regression.get('r'),
            'F_cum': regression.get('F'),
            'E_cum': regression.get('E'),
            'T_cum': regression.get('T'),
            'incumbent_id': self._inc_id,
            'iters_as_incumbent': iter - self._inc_since_iter + 1,
            'regime_triggered': False if stopped else self._last_sampling['regime_triggered'],
            'n_regressed_in_batch': 0 if stopped else self._last_sampling['n_regressed_in_batch'],
        }
        if stopped:
            payload['stopped'] = stopped
        trace(self.logger, 'iteration', payload)

    def _trace_trigger(self, iter, node, regression):
        trace(self.logger, 'regression_trigger', {
            'iter': iter,
            'node_id': node.id,
            'incumbent_id': self._inc_id,
            'r_cum': regression.get('r'),
            'T_cum': regression.get('T'),
            'F_cum': regression.get('F'),
            'E_cum': regression.get('E'),
            'iters_as_incumbent': iter - self._inc_since_iter + 1,
            'n_regressed_keys': len(regression.get('regressed_keys', [])),
            'n_regressed_in_batch': self._last_sampling['n_regressed_in_batch'],
        })

    def _trace_candidate(self, iter, parent, cand, kind, val_acc, prev_max_acc, accepted, source_node=None):
        payload = {
            'iter': iter,
            'parent_id': parent.id,
            'node_id': cand.id,
            'type': kind,
            'val_acc': float(val_acc),
            'prev_max_acc': float(prev_max_acc),
            'accepted': bool(accepted),
        }
        if source_node is not None:
            # SIMP simplifica o incumbente, que pode diferir do nó pai registrado
            payload['source_node_id'] = source_node.id
        trace(self.logger, 'candidate', payload)


    
    def step(self, node:GraceNode):
        new_nodes, gradient_descent_output = self._gradient_descent_step(node=node)
        return new_nodes, gradient_descent_output
    
    def build_root(self, init_prompt):
        # Calculate curriculum for the training set
        if self.curriculum_mode == 'static':
            self._calculate_curriculum(init_prompt)
        elif self.curriculum_mode == 'bayes':
            self._init_tracker(init_prompt)
        # 'off': GRACE original, sem passada zero-shot

        node = GraceNode(prompt=init_prompt, action=None, parent=None,mom_prompt=None)

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

    def _init_tracker(self, init_prompt):
        """
        curriculum_mode='bayes': atipicidade intrínseca + resultados zero-shot de P0
        -> DifficultyTracker.register.
        """
        self.logger.info("\n" + "------"*40)
        self.logger.info("INITIALIZING BAYESIAN DIFFICULTY TRACKER")
        self.logger.info("-----"*40)

        examples = []
        for batch in self.train_dataloader:
            for q, a in zip(batch['question'], batch['answer']):
                examples.append({'question': q, 'answer': a})

        classifier = CurriculumClassifier(
            task=self.task,
            base_model=self.base_model,
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            logger=self.logger
        )
        # Só para obter classifier.embeddings; as distâncias euclidianas são ignoradas.
        classifier.calculate_semantic_complexity(examples)
        embeddings = classifier.embeddings
        failed_mask = classifier.evaluate_baseline_performance(examples, init_prompt)
        p0_correct = (~np.asarray(failed_mask, dtype=bool)).astype(int)

        labels = [str(l) for l in self.task.clean_labels([ex['answer'] for ex in examples])]

        if self.intrinsic_mode == 'hdbscan':
            z, typical, diag = DifficultyTracker.atypicality_hdbscan(
                embeddings, labels,
                centroid_mode=self.centroid_mode,
                typical_percentile=self.typical_percentile,
                min_group_size=self.min_group_size,
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
            )
        else:
            z, typical, diag = DifficultyTracker.atypicality_centroid(
                embeddings, labels,
                centroid_mode=self.centroid_mode,
                typical_percentile=self.typical_percentile,
                min_group_size=self.min_group_size,
            )

        trace(self.logger, 'init_groups', {
            'intrinsic_mode': self.intrinsic_mode,
            'centroid_mode': self.centroid_mode,
            'groups': diag,
        })
        self.tracker.register([ex['question'] for ex in examples], z, typical, p0_correct)
        self.tracker.log_state('init', with_items=True)
        self.logger.info("✅ Difficulty tracker initialized.")

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
                                           temperature=self.eval_temperature,
                                           )
        return metric, eval_output
    


