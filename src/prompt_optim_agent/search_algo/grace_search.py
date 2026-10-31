import itertools
from typing import Generic, Optional, List
from .base_algo import SearchAlgo, State, Action
import json
import os

import global_vars

class GraceNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, 
                 prompt: str, 
                 action: str = None,
                 parent: "Optional[GraceNode]" = None,
                 mom_prompt: str = None,
                 
                 ):

        self.id = next(GraceNode.id_iter)
        self.prompt = prompt
        self.mom_prompt = mom_prompt
        self.test_metric = -1.0
        self.eval_metric = 0. 
        self.action = action
        self.parent = parent
        self.children: 'Optional[list[GraceNode]]' = []
        self.batch = None
        
        # Momentum: controla dificuldade dos exemplos baseado na performance
        self.train_accuracy = 0.0  # Acurácia no conjunto de treino
        self.momentum = 'EASY'  # EASY, MEDIUM, HARD
        # curriculum_mode='bayes': estatística r do teste de regressão (só para log)
        self.regression_r = None
        # acceptance_mode='evidence': True se o candidato foi rejeitado cedo e
        # eval_metric é uma ESTIMATIVA (avaliação parcial); ver docs/lose_to_win.md
        self.eval_estimated = False
        self.eval_n_items = None

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
    
    def to_dict(self):
        if self.parent is None:
            p_id = -1
        else:
            p_id = self.parent.id
        
        d = {
            'id': self.id,
            'action':self.action,
            'depth':self.depth,
            'parent':p_id,
            'eval_metric': self.eval_metric,
            'test_metric': self.test_metric,
            'prompt':self.prompt,
            'train_accuracy': self.train_accuracy,
            'momentum': self.momentum,
            'regression_r': self.regression_r
        }
        if self.eval_estimated:
            # só aparece em acceptance_mode='evidence' (default: results.json inalterado)
            d['eval_estimated'] = True
            d['eval_n_items'] = self.eval_n_items
        return d

class GraceSearch(SearchAlgo):

    def __init__(
        self, 
        task,
        world_model, 
        
        # log
        logger=None,
        log_dir = None,

        # seleção final e teste; None = usa o valor do world model (vem do YAML,
        # world_model_setting)
        final_selection: Optional[str] = None,       # 'incumbent' | 'legacy_ties'
        test_root: Optional[bool] = None,            # testa também o prompt raiz
        test_all_incumbents: Optional[bool] = None,  # testa cada incumbente da cadeia
        **kwargs,
        ) -> None:

        self.task = task
        self.world_model = world_model
        self.logger = logger
        self.log_dir = log_dir

        def pick(value, name, default):
            return value if value is not None else getattr(world_model, name, default)

        self.final_selection = pick(final_selection, 'final_selection', 'incumbent')
        self.test_root = bool(pick(test_root, 'test_root', True))
        self.test_all_incumbents = bool(pick(test_all_incumbents, 'test_all_incumbents', False))
        assert self.final_selection in ('incumbent', 'legacy_ties'), \
            f"final_selection inválido: {self.final_selection}"

        self.nodes:List[GraceNode] = [] 
        self.all_nodes:List[GraceNode] = []
        
        self.log_vars()
    
    def log_vars(self):
        self.logger.info('-------------------- GRACE Search -----------------------')
        ignored_print_vars = ['nodes', 'all_nodes']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            self.logger.info(f'{var_name} : {var_value}')
        self.logger.info('-------------------------------------------')

    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
        
    def search(self, init_state: str, **kwargs):
        
        self.root = self.world_model.build_root(init_state)
        self.all_nodes.append(self.root)
    
        nodes = [self.root]
        self.nodes = nodes
        

        self.logger.info(f'----------------  Iteration Begin ----------------')
        self.nodes[-1].batch = {"question":[],'answer':[]}
        new_nodes, _ = self.world_model.step(self.nodes[-1])
        self.all_nodes.extend(new_nodes)
        self.nodes = new_nodes
        
        output = self.prepare_output()
        self.output_to_json(output=output)
        
        return self.nodes, output

    def __call__(self, init_state: str, **kwargs):
        GraceNode.reset_id()

        nodes, output = self.search(init_state=init_state)
        
        return nodes, output 


    def prepare_output(self):
        # test and log nodes
        self.logger.info(f'\n---------------------  test nodes ------------------------')
        

        def get_latest_eval_metric(node):
            metric = node.eval_metric
            return metric[-1] if isinstance(metric, tuple) else metric

        by_id = {node.id: node for node in self.all_nodes}
        incumbent_ids = [i for i in getattr(self.world_model, 'incumbent_ids', []) if i in by_id]
        chain = [by_id[i] for i in incumbent_ids]

        selection = self.final_selection
        if selection == 'incumbent' and not chain:
            self.logger.info("AVISO: cadeia de incumbentes vazia; usando final_selection='legacy_ties'.")
            selection = 'legacy_ties'

        selected_node_id = None
        if selection == 'incumbent':
            # Só nós que foram incumbentes têm eval exato garantido. Maior eval_metric;
            # empate -> o primeiro da cadeia (max devolve o 1º máximo).
            selected = max(chain, key=get_latest_eval_metric)
            selected_node_id = selected.id
            to_test = [selected]
        else:
            scored_nodes = [(node, get_latest_eval_metric(node)) for node in self.all_nodes]
            max_metric = max(score for _,score in scored_nodes)
            to_test = [node for node, score in scored_nodes if score == max_metric]

        if self.test_root:
            to_test.append(self.root)
        if self.test_all_incumbents:
            to_test.extend(chain)

        tested = {}
        for node in to_test:
            if node.id in tested:
                continue
            node.test_metric, _ = self.world_model.test_prompt(node.prompt, node_id=node.id)
            tested[node.id] = node

        cost_dic = {"base_api_count":global_vars.base_api_count,"base_input_token":global_vars.base_input_token,"base_output_token":global_vars.base_output_token,
                    "target_api_count":global_vars.target_api_count,"target_input_token":global_vars.target_input_token,"target_output_token":global_vars.target_output_token,}

        return dict(
            all_nodes = self.all_nodes,
            cost = cost_dic,
            final_selection = selection,
            selected_node_id = selected_node_id,   # None em 'legacy_ties' (vários empatados testados)
            tested_node_ids = list(tested),
            root_test_metric = self.root.test_metric if self.test_root else None,
            incumbent_ids = incumbent_ids,
        )


    def output_to_json(self, output):
        data_to_save = {}

        for key in output:
            if key == "all_nodes":
                data_to_save[key] = [node.to_dict() for node in output[key]]
            else:
                data_to_save[key] = output[key]
        with open(os.path.join(self.log_dir, 'results.json'), 'w') as f:
            json.dump(data_to_save, f, indent=4)




    