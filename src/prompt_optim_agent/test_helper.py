import os
from tqdm import tqdm
from .world_model.prompts import *
from tasks import *
from .utils import *
import logging


def eval_instruction_with_loader(task, eval_prompt, base_model, dataloader,  temperature=0, record_outputs=True, logger=None):
    '''
        evaluate cur_prompt on task testing dataset
    '''
    
    build_forward_prompts_func = task.build_forward_prompts_completion
    batch_forward_func = base_model.batch_forward_func
    
    all_questions = []
    all_labels = []
    all_preds = []
    all_prompts = []
    all_responses = []
    eval_output = {}
    
    # Usar logger padrão se não fornecido
    if logger is None:
        logger = logging.getLogger(__name__)
    
    pbar = tqdm(dataloader, leave=False)
    batch_idx = 0
    
    for batch in pbar:
        batch_prompts = build_forward_prompts_func(batch['question'], eval_prompt)
        
        # Log input para cada exemplo no batch
        for i, prompt in enumerate(batch_prompts):
            logger.info(f"---------------\t\t{batch_idx * len(batch_prompts) + i}\t\t----------------")
            logger.info(f"Input:\n{prompt}")
        
        responses = batch_forward_func(batch_prompts)
        
        # Log output para cada exemplo no batch
        for i, response in enumerate(responses):
            logger.info(f"Output:\n{response}")
            logger.info(f"---------------\t\t\t\t----------------")
        
        preds = task.batch_clean_responses(responses)
        labels = task.clean_labels(batch['answer'])
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_questions.extend(batch['question'])
        if record_outputs:
            all_prompts.extend(batch_prompts)
            all_responses.extend(responses)
        metric = task.cal_metric(all_preds, all_labels, all_questions)
        if not isinstance(metric, tuple):
            pbar.set_postfix_str(f"Test Metric: {metric:.4f}")
        else:
            pbar.set_postfix_str(f"Test Metrics: {metric}")
        
        batch_idx += 1
    
    if record_outputs:
        eval_output['model_inputs'] =  all_prompts
        eval_output['model_responses'] =  all_responses
        eval_output['preds'] =  all_preds
        eval_output['labels'] =  all_labels
    eval_output['correct'] =  task.cal_correct(all_preds, all_labels)    
    metric = task.cal_metric(all_preds, all_labels, all_questions)
    return metric, eval_output
    
