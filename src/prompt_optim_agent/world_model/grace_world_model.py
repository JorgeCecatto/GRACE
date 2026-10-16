from .gradient_descent import *
from ..test_helper import eval_instruction_with_loader
from typing import Generic
from ..search_algo.base_algo import State, Action
from ..search_algo.grace_search import GraceNode
from ..curriculum.curriculum_classifier import CurriculumClassifier
from ..curriculum.difficulty_tracker import DifficultyTracker, trace
from .evidence_gate import EvidenceGate

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
        eval_temperature: float = 0.0,  # avaliacao deterministica por padrao (reprodutibilidade)

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

        # aceitação por evidência "Lose to Win" (docs/lose_to_win.md)
        acceptance_mode: str = 'strict',      # 'strict' | 'evidence_shadow' | 'evidence'
        evidence_margin: int = 1,             # m: aceita sse Delta >= m (m=1 == regra atual)
        evidence_delta: float = 0.05,         # rejeita cedo se P <= delta; completa se P >= 1-delta
        evidence_kappa: float = 5.0,          # força do prior empírico agrupado
        evidence_warmup: int = 10,            # candidatos OPT avaliados por inteiro antes de usar o histórico
        evidence_mc_samples: int = 2000,      # amostras Monte Carlo para P(Delta_total >= m)
        evidence_batch_size: int = None,      # tamanho do lote sequencial; None = eval_batch_size
        evidence_seed: int = 0,               # seed do Monte Carlo (rng próprio)

        # ordenação do treino por Thompson (só tem efeito com curriculum_mode='bayes')
        train_order: str = 'default',         # 'default' | 'thompson'
        train_reserve_random: int = 2,        # posições aleatórias por lote (thompson)
        no_fixable_action: str = 'log',       # 'log' | 'simp' | 'stop' (thompson)
        fixable_mass_threshold: float = 3.0,  # limiar da soma de P(erro) dos não irredutíveis
        **kwargs
        ) -> None:

        assert acceptance_mode in ('strict', 'evidence_shadow', 'evidence'), f"acceptance_mode inválido: {acceptance_mode}"
        assert train_order in ('default', 'thompson'), f"train_order inválido: {train_order}"
        assert no_fixable_action in ('log', 'simp', 'stop'), f"no_fixable_action inválido: {no_fixable_action}"
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
        # Acertos por item (0/1, ordem fixa do eval) da avaliação que gerou o max_acc
        # do incumbente atual. Sempre mantido (instrumentação); base do teste pareado.
        self._inc_eval_correct = None

        # Thompson (train_order='thompson'): só com curriculum_mode='bayes'
        self.train_order = train_order
        self.train_reserve_random = int(train_reserve_random)
        self.no_fixable_action = no_fixable_action
        self.fixable_mass_threshold = float(fixable_mass_threshold)
        self._thompson = (train_order == 'thompson' and curriculum_mode == 'bayes')
        self._train_items = None          # [{'question', 'answer'}], criado sob demanda
        self._irreducible = set()         # chaves marcadas como irredutíveis
        self._fail_incs = {}              # chave -> conjunto de incumbentes sob os quais errou
        self._simp_armed = True           # no_fixable_action='simp': rearmado quando um OPT é aceito

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

        self.train_batch_size = train_batch_size
        if train_order == 'thompson' and curriculum_mode != 'bayes':
            self.logger.info("train_order='thompson' ignorado: requer curriculum_mode='bayes'.")

        # Aceitação por evidência (strict = comportamento original, sem gate)
        self.acceptance_mode = acceptance_mode
        self.gate = None
        self._n_eval = len(self.eval_dataloader.dataset)
        self._evidence_lot_size = int(evidence_batch_size) if evidence_batch_size else int(eval_batch_size)
        if acceptance_mode != 'strict':
            # A métrica de aceitação precisa ser a acurácia (média de 'correct'); o NCBI usa F1.
            assert "ncbi" not in self.task.task_name.lower(), \
                "acceptance_mode != 'strict' não é suportado no NCBI (a métrica de aceitação é F1, não acurácia)."
            self.gate = EvidenceGate(
                n_items=self._n_eval,
                margin=evidence_margin,
                risk=evidence_delta,
                kappa=evidence_kappa,
                warmup=evidence_warmup,
                mc_samples=evidence_mc_samples,
                seed=evidence_seed,
            )
            if self._evidence_lot_size >= self._n_eval:
                self.logger.info(
                    f"AVISO: lote sequencial ({self._evidence_lot_size}) >= itens do eval ({self._n_eval}): "
                    "há um único lote por candidato e nenhuma rejeição precoce. "
                    "Defina evidence_batch_size menor.")

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
        n_batches = 0

        # train_order='thompson': lotes montados por amostragem de Thompson (lazy: o
        # break abaixo interrompe o gerador). Default: o dataloader original.
        batches = self._thompson_batches() if self._thompson else self.train_dataloader

        for batch in batches:
            n_batches += 1
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

            if self._thompson:
                self._update_irreducible(observations)
                aggregated_output['fixable'] = self._fixable_mass()

        n_examples = len(aggregated_output['examples'])
        n_right = sum(1 for c in aggregated_output['correct'] if c)
        trace(self.logger, 'train_cost', {
            'n_examples': n_examples,
            'n_batches': n_batches,
            'n_wrong': n_examples - n_right,
            'n_right': n_right,
        })

        return aggregated_output

    # ------------------------------------------------------------------ Thompson
    def _get_train_items(self):
        """Exemplos de treino [{'question', 'answer'}] (mesma chave do DifficultyTracker)."""
        if self._train_items is None:
            items = []
            for batch in self.train_dataloader:
                for q, a in zip(batch['question'], batch['answer']):
                    items.append({'question': q, 'answer': a})
            self._train_items = items
        return self._train_items

    def _thompson_batches(self):
        """
        Gera lotes de treino ordenados por Thompson. Para cada exemplo sorteia
        theta_i ~ Beta(kappa*m_i + s_i, kappa*(1-m_i) + n_i - s_i) (parâmetros do
        DifficultyTracker.beta_params) e ordena por theta crescente (mais chance de
        erro primeiro), excluindo os irredutíveis. Cada lote tem train_reserve_random
        posições aleatórias (sobre todos os exemplos ainda não usados, inclusive os
        irredutíveis) e o restante tomado da ordem de Thompson.
        """
        items = self._get_train_items()
        n = len(items)
        rng = self.curriculum_rng
        params = [self.tracker.beta_params(it['question']) for it in items]
        theta = rng.beta(np.array([p[0] for p in params]), np.array([p[1] for p in params]))
        order = [int(i) for i in np.argsort(theta, kind='stable')
                 if items[int(i)]['question'] not in self._irreducible]

        bs = self.train_batch_size
        reserve = min(self.train_reserve_random, bs)
        used = set()
        ptr = 0
        while len(used) < n:
            batch_idx = []
            avail = [i for i in range(n) if i not in used]
            r = min(reserve, len(avail))
            if r > 0:
                for i in rng.choice(avail, size=r, replace=False):
                    used.add(int(i))
                    batch_idx.append(int(i))
            while len(batch_idx) < bs and ptr < len(order):
                i = order[ptr]
                ptr += 1
                if i in used:
                    continue
                used.add(i)
                batch_idx.append(i)
            if not batch_idx:
                break
            yield {
                'question': [items[i]['question'] for i in batch_idx],
                'answer': [items[i]['answer'] for i in batch_idx],
            }

    @staticmethod
    def _beta_cdf(x, a, b, rng, n_mc=4000):
        """P(Beta(a, b) < x): scipy se disponível; senão Monte Carlo."""
        try:
            from scipy.special import betainc
            return float(betainc(a, b, x))
        except ImportError:
            return float((rng.beta(a, b, size=n_mc) < x).mean())

    # Critério de irredutível (train_order='thompson')
    _IRRED_MIN_INCUMBENTS = 3   # errou sob >= 3 incumbentes DIFERENTES
    _IRRED_P_LOW = 0.1          # ... e P(p_i < 0.1) ...
    _IRRED_CONF = 0.9           # ... > 0.9 pelo posterior

    def _update_irreducible(self, observations):
        """
        Marca/reabilita irredutíveis a partir das observações desta chamada.
        Marca: errou sob >= 3 incumbentes diferentes E P(p_i < 0.1) > 0.9.
        Reabilita: acertou alguma vez (limpa também o histórico de incumbentes).
        """
        for k, c, _ in observations:
            if c:
                self._fail_incs.pop(k, None)
                if k in self._irreducible:
                    self._irreducible.discard(k)
                    trace(self.logger, 'irreducible', {
                        'action': 'rehabilitate', 'q': k[:80], 'incumbent_id': self._inc_id,
                    })
            else:
                self._fail_incs.setdefault(k, set()).add(self._inc_id)

        for k in dict.fromkeys(k for k, c, _ in observations if not c):
            if k in self._irreducible:
                continue
            n_inc = len(self._fail_incs.get(k, ()))
            if n_inc < self._IRRED_MIN_INCUMBENTS:
                continue
            a, b = self.tracker.beta_params(k)
            p_low = self._beta_cdf(self._IRRED_P_LOW, a, b, self.curriculum_rng)
            if p_low > self._IRRED_CONF:
                self._irreducible.add(k)
                trace(self.logger, 'irreducible', {
                    'action': 'mark', 'q': k[:80], 'incumbent_id': self._inc_id,
                    'n_incumbents_failed': n_inc, 'p_below': p_low,
                })

    def _fixable_mass(self):
        """
        Sinal de "melhor esforço": soma de P(erro) = 1 - p_hat dos exemplos de treino
        NÃO irredutíveis. Soma baixa = quase nada ainda corrigível pelo treino.
        """
        mass = 0.0
        n_active = 0
        for k in self.tracker.items:
            if k in self._irreducible:
                continue
            mass += 1.0 - self.tracker.p_success(k)
            n_active += 1
        out = {'mass': float(mass), 'n_active': n_active, 'n_irreducible': len(self._irreducible)}
        trace(self.logger, 'fixable_mass', {
            **out,
            'threshold': self.fixable_mass_threshold,
            'action': self.no_fixable_action,
            'incumbent_id': self._inc_id,
        })
        return out

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
        # Instrumentação: índices estáveis do eval, custo e acertos por item da raiz
        trace(self.logger, 'eval_index', {
            'n_items': len(eval_batch['question']),
            'items': [{'idx': j, 'q': q[:80]} for j, q in enumerate(eval_batch['question'])],
        })
        self._inc_eval_correct = self._correct_vector(eval_forward_output)
        self._trace_eval(child_node, None, 'ROOT', None, eval_forward_output)
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

            # no_fixable_action (train_order='thompson'): sinal de melhor esforço.
            # 'log' só registra (o trace 'fixable_mass' já foi emitido no train_forward).
            fixable = train_forward_output.get('fixable')
            if (fixable is not None and self.no_fixable_action != 'log'
                    and fixable['mass'] < self.fixable_mass_threshold):
                if self.no_fixable_action == 'stop':
                    # Encerra a otimização deste passo.
                    trace(self.logger, 'fixable_action', {
                        'action': 'stop', 'iter': iter, 'mass': fixable['mass'], 'node_id': child_node.id,
                    })
                    self._trace_iteration(iter, child_node, cur_acc, regression, regression_iter,
                                          stopped='no_fixable_stop')
                    break
                if self.no_fixable_action == 'simp' and self._simp_armed:
                    # Dispara o SIMP no lugar do OPT desta iteração. Fica desarmado até
                    # um OPT ser aceito, para não simplificar a cada iteração seguida.
                    self._simp_armed = False
                    trace(self.logger, 'fixable_action', {
                        'action': 'simp', 'iter': iter, 'mass': fixable['mass'], 'node_id': child_node.id,
                    })
                    self._trace_iteration(iter, child_node, cur_acc, regression, regression_iter,
                                          stopped='no_fixable_simp')
                    child_node, max_acc, eval_forward_output = self._run_simp(
                        iter, cur_child_node, child_node, eval_batch, eval_forward_output, max_acc, new_nodes)
                    stop_early = 0
                    continue

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
                incumbent_node = child_node  # incumbente contra o qual este candidato é avaliado
                eval_temp_forward_output, ev = self._eval_candidate(opt_prompt, eval_batch)
                temp_child_node = GraceNode(
                    prompt=opt_prompt,
                    action="OPT",
                    mom_prompt=None,
                    parent=cur_child_node,
                    )
                temp_child_node.eval_metric = (eval_forward_output['acc'],eval_temp_forward_output['acc'])
                new_nodes.append(temp_child_node)
                estimated = bool(ev and ev.get('rejected_early'))
                if estimated:
                    # avaliação parcial: eval_metric é ESTIMATIVA (nunca entra no max_acc)
                    temp_child_node.eval_estimated = True
                    temp_child_node.eval_n_items = ev['n_evaluated']
                self._trace_eval(temp_child_node, cur_child_node, 'OPT', incumbent_node,
                                 eval_temp_forward_output, ev)

                cand_acc = self._sort_helper(eval_temp_forward_output['acc'])
                if ev is not None and 'accepted' in ev:
                    accepted = ev['accepted']   # acceptance_mode='evidence': Delta >= m
                else:
                    accepted = cand_acc > max_acc
                self._trace_shadow(temp_child_node, cur_child_node, incumbent_node, ev, accepted)
                if self.curriculum_mode == 'bayes':
                    self._trace_candidate(iter, cur_child_node, temp_child_node, 'OPT',
                                          cand_acc, max_acc,
                                          accepted=accepted, estimated=estimated,
                                          n_items=ev['n_evaluated'] if estimated else None)

                #Change current prompt
                if accepted:
                    stop_early = 0
                    self._simp_armed = True
                    # Só chega aqui com avaliação completa: cand_acc é exato.
                    max_acc = cand_acc
                    child_node = temp_child_node
                    self._inc_eval_correct = self._correct_vector(eval_temp_forward_output)
                    if self.curriculum_mode == 'bayes':
                        self._set_incumbent(child_node, iter + 1)
                    #Re-Split train into correct and wrong samples
                    # train_forward_output = self.train_forward(cur_prompt=opt_prompt)
                    eval_forward_output = eval_temp_forward_output
                else:
                    stop_early+=1

            if stop_early==self.stop_early_thresh:
                child_node, max_acc, eval_forward_output = self._run_simp(
                    iter, cur_child_node, child_node, eval_batch, eval_forward_output, max_acc, new_nodes)
                stop_early = 0

        if self.gate is not None:
            trace(self.logger, 'evidence_summary', self.gate.summary())
        if self.curriculum_mode == 'bayes':
            self.tracker.log_state('step_end', with_items=True)

        return new_nodes, None

    def _run_simp(self, iter, cur_child_node, child_node, eval_batch, eval_forward_output, max_acc, new_nodes):
        """
        Passo SIMP: simplifica o incumbente, avalia POR INTEIRO e sempre aceita
        (substitui o incumbente, em todos os acceptance_mode). Devolve
        (novo incumbente, novo max_acc, eval_forward_output do novo incumbente).
        """
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
        self._trace_eval(temp_child_node, cur_child_node, 'SIMP', child_node, eval_temp_forward_output)
        if self.curriculum_mode == 'bayes':
            # SIMP é sempre aceito pelo GRACE (substitui o incumbente)
            self._trace_candidate(iter, cur_child_node, temp_child_node, 'SIMP',
                                  self._sort_helper(eval_temp_forward_output['acc']), max_acc,
                                  accepted=True, source_node=child_node)

        max_acc = self._sort_helper(eval_temp_forward_output['acc'])
        child_node = temp_child_node
        self._inc_eval_correct = self._correct_vector(eval_temp_forward_output)
        if self.curriculum_mode == 'bayes':
            self._set_incumbent(child_node, iter + 1)
        # train_forward_output = self.train_forward(cur_prompt=opt_prompt)
        return child_node, max_acc, eval_temp_forward_output

    # ---------------------------------------------------- avaliação de candidatos
    @staticmethod
    def _correct_vector(forward_output):
        """Acertos por item (0/1) na ordem fixa do eval."""
        return np.asarray([int(c) for c in forward_output['correct']], dtype=int)

    def _eval_candidate(self, opt_prompt, eval_batch):
        """
        Avalia um candidato OPT no eval conforme acceptance_mode. Devolve
        (forward_output, ev); ev é None em 'strict'.
        - strict: forward completo, como sempre.
        - evidence_shadow: forward completo (decisão = strict) + simulação do que a
          regra sequencial teria feito (ev['shadow']).
        - evidence: aquecimento = avaliação completa; depois, avaliação sequencial com
          rejeição precoce. ev['accepted'] decide (Delta >= m).
        """
        if self.acceptance_mode == 'strict':
            return self.gradient_descent.forward(
                batch=eval_batch, cur_prompt=opt_prompt, temperature=self.eval_temperature), None

        gate = self.gate
        inc = np.asarray(self._inc_eval_correct, dtype=int)
        assert len(inc) == len(eval_batch['question']) == gate.n, "tamanho do eval inconsistente com o gate"
        warm = gate.in_warmup()

        if self.acceptance_mode == 'evidence' and not warm:
            def fetch_lot(idx):
                sub = {'question': [eval_batch['question'][j] for j in idx],
                       'answer': [eval_batch['answer'][j] for j in idx]}
                out = self.gradient_descent.forward(
                    batch=sub, cur_prompt=opt_prompt, temperature=self.eval_temperature)
                return [int(c) for c in out['correct']]

            res = gate.run(inc, fetch_lot, self._evidence_lot_size)
            cand = res['cand_correct']
            gate.update(inc, cand)
            n = len(cand)
            if res['rejected_early']:
                # ESTIMATIVA: acerto esperado do incumbente + Delta esperado até o fim,
                # limitada a ficar abaixo da acurácia do incumbente, para que um
                # candidato rejeitado nunca vire "melhor nó" em prepare_output.
                est = (int(inc.sum()) + res['est_delta_total']) / n
                acc = max(0.0, min(est, (int(inc.sum()) - 1) / n))
                res['raw_estimate'] = float(est)
            else:
                acc = np.mean(cand.tolist())
            res['warmup'] = False
            return {'cur_prompt': opt_prompt, 'correct': cand.tolist(), 'acc': acc}, res

        # strict-equivalente: avaliação completa (aquecimento, ou modo sombra)
        out = self.gradient_descent.forward(
            batch=eval_batch, cur_prompt=opt_prompt, temperature=self.eval_temperature)
        cand = self._correct_vector(out)
        delta = int(((cand == 1) & (inc == 0)).sum() - ((cand == 0) & (inc == 1)).sum())
        ev = {'warmup': warm, 'delta_full': delta}
        if self.acceptance_mode == 'evidence':
            ev['accepted'] = delta >= gate.margin
        else:
            ev['shadow'] = None if warm else gate.run(
                inc, lambda idx: cand[idx], self._evidence_lot_size, record_all=True)
        gate.update(inc, cand)
        return out, ev

    def _trace_eval(self, node, parent, kind, incumbent, forward_output, ev=None):
        """Eventos 'eval_cost' e 'eval_items' (acertos por item na ordem fixa do eval)."""
        if ev is not None and 'cand_correct' in ev:
            vec = [None if c < 0 else int(c) for c in ev['cand_correct']]  # None = não avaliado
            n_eval = ev['n_evaluated']
        else:
            vec = [int(c) for c in forward_output['correct']]
            n_eval = len(vec)
        trace(self.logger, 'eval_cost', {
            'node_id': node.id, 'n_items_avaliados': n_eval, 'n_items_total': len(vec),
        })
        trace(self.logger, 'eval_items', {
            'node_id': node.id,
            'parent_id': parent.id if parent is not None else None,
            'type': kind,
            'incumbent_id': incumbent.id if incumbent is not None else None,
            'correct': vec,
        })

    def _trace_shadow(self, node, parent, incumbent, ev, actual_accepted):
        """Modo sombra: o que a regra sequencial teria decidido (a decisão real foi a strict)."""
        if ev is None or 'shadow' not in ev:
            return
        sim = ev['shadow']
        payload = {
            'node_id': node.id,
            'parent_id': parent.id,
            'incumbent_id': incumbent.id,
            'warmup': ev['warmup'],
            'actual_accepted': bool(actual_accepted),
            'delta_full': ev['delta_full'],
            'n_items_total': self._n_eval,
        }
        if sim is None:
            payload['hypothetical'] = None   # aquecimento: sem decisão hipotética
        else:
            payload.update({
                'hypothetical': 'reject_early' if sim['rejected_early']
                                else ('accept' if sim['accepted'] else 'reject_full'),
                'hypothetical_accepted': sim['accepted'],
                'n_items_used': sim['n_evaluated'],
                'agrees': sim['accepted'] == bool(actual_accepted),
                'steps': sim['steps'],   # por lote: n_items, delta parcial, P, ação
            })
        trace(self.logger, 'evidence_shadow', payload)

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

    def _trace_candidate(self, iter, parent, cand, kind, val_acc, prev_max_acc, accepted, source_node=None,
                         estimated=False, n_items=None):
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
        if estimated:
            # val_acc é ESTIMATIVA (rejeição precoce, acceptance_mode='evidence')
            payload['eval_estimated'] = True
            payload['n_items_evaluated'] = n_items
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
        trace(self.logger, 'init_cost', {'n_items': len(examples)})

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

    def test_prompt(self, prompt, node_id=None):
        metric, eval_output = eval_instruction_with_loader(task=self.task,
                                           eval_prompt=prompt,
                                           dataloader=self.test_dataloader,
                                           base_model=self.base_model,
                                           temperature=self.eval_temperature,
                                           )
        trace(self.logger, 'test_cost', {'node_id': node_id, 'n_items': len(eval_output['correct'])})
        return metric, eval_output
    


