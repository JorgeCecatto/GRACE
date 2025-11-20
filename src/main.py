import argparse
from prompt_optim_agent import *
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config():
    parser = argparse.ArgumentParser(description='Process prompt search agent arguments')

    parser.add_argument('--config_dir', type=str, default='./defualt_config.yaml')
   
    args = parser.parse_args()

    args = load_config(args.config_dir)
    return args

def validate_config(config):
    # Basic settings
    assert config['task_name'] is not None, "task_name must be specified"
    assert config['search_algo'] in ['grace'], "search_algo must be 'grace'"
    assert isinstance(config['print_log'], bool), "print_log must be a boolean"
    assert config['log_dir'] is not None, "log_dir must be specified"
    assert config['init_prompt'] is not None, "init_prompt must be specified"

    # Task setting
    assert isinstance(config['task_setting']['train_size'], (int, type(None))), "train_size must be an integer or None"
    assert isinstance(config['task_setting']['eval_size'], int), "eval_size must be an integer"
    assert isinstance(config['task_setting']['test_size'], int), "test_size must be an integer"
    assert isinstance(config['task_setting']['seed'], int), "seed must be an integer"
    assert isinstance(config['task_setting']['post_instruction'], bool), "post_instruction must be a boolean"
    # Base model setting
    assert config['base_model_setting']['model_type'] in ['openai', 'ollama'], \
        "base_model.model_type must be 'openai' or 'ollama'"
    assert config['base_model_setting']['model_name'] is not None, "base_model.model_name must be specified"
    assert isinstance(config['base_model_setting']['temperature'], float), "base_model.temperature must be a float"
    
    if config['base_model_setting']['model_type'] == 'openai' and config['base_model_setting']['api_key'] is None:
        raise ValueError("Please set base model's api key")
    assert isinstance(config['base_model_setting']['base_model'], bool), "base_model must be a boolean"

    # Optim model setting
    assert config['optim_model_setting']['model_type'] in ['openai', 'ollama'], \
        "optim_model.model_type must be 'openai' or 'ollama'"
    assert config['optim_model_setting']['model_name'] is not None, "optim_model.model_name must be specified"
    assert isinstance(config['optim_model_setting']['temperature'], float), "optim_model.temperature must be a float"
   
    if config['optim_model_setting']['model_type'] == 'openai' and config['optim_model_setting']['api_key'] is None:
        raise ValueError("Please set optim model's api key")
    assert isinstance(config['optim_model_setting']['base_model'], bool), "base_model must be a boolean"
    
    # Meta model setting (optional - usado apenas se use_meta_prompting=true)
    if 'meta_model_setting' in config:
        assert config['meta_model_setting']['model_type'] in ['openai', 'ollama'], \
            "meta_model.model_type must be 'openai' or 'ollama'"
        assert config['meta_model_setting']['model_name'] is not None, "meta_model.model_name must be specified"
        assert isinstance(config['meta_model_setting']['temperature'], float), "meta_model.temperature must be a float"
        
        if config['meta_model_setting']['model_type'] == 'openai' and config['meta_model_setting']['api_key'] is None:
            raise ValueError("Please set meta model's api key")
    
    # World model setting
    assert isinstance(config['world_model_setting']['iteration_num'], int), "search.iteration_num must be an integer"
    assert isinstance(config['world_model_setting']['stop_early_thresh'], int), "world_model.stop_early_thresh must be an integer"
    assert isinstance(config['world_model_setting']['num_correct_sample'], int), "world_model.num_correct_sample must be an integer"
    assert isinstance(config['world_model_setting']['num_wrong_sample'], int), "world_model.num_wrong_sample must be an integer"

    assert isinstance(config['world_model_setting']['num_new_prompts'], int), "world_model.num_new_prompts must be an integer"

    assert isinstance(config['world_model_setting']['train_shuffle'], bool), "world_model.train_shuffle must be a boolean"

    if 'use_meta_prompting' in config['world_model_setting']:
        assert isinstance(config['world_model_setting']['use_meta_prompting'], bool), \
            "use_meta_prompting must be a boolean"
        
        if config['world_model_setting']['use_meta_prompting']:
            # Strategy
            valid_strategies = ['expert_generalist', 'task_decomposition', 'zero_shot_cot']
            assert config['world_model_setting'].get('meta_prompt_strategy', 'expert_generalist') in valid_strategies, \
                f"meta_prompt_strategy must be one of {valid_strategies}"
            
            # Num candidates
            assert isinstance(config['world_model_setting'].get('meta_prompt_num_candidates', 3), int), \
                "meta_prompt_num_candidates must be an integer"
            assert 1 <= config['world_model_setting'].get('meta_prompt_num_candidates', 3) <= 5, \
                "meta_prompt_num_candidates must be between 1 and 5"
            
            # Eval samples
            assert isinstance(config['world_model_setting'].get('meta_prompt_eval_samples', 10), int), \
                "meta_prompt_eval_samples must be an integer"
            assert config['world_model_setting'].get('meta_prompt_eval_samples', 10) >= 5, \
                "meta_prompt_eval_samples must be at least 5"
    


def main(args):
    agent = BaseAgent(**args)
    agent.run()
    return

if __name__ == '__main__':
    args = config()
    validate_config(args)
    print(args)
    main(args)