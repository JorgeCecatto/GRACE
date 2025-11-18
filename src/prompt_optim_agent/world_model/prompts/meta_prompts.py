zs_prompt = """You are an expert at adding reasoning to instructions.

Current instruction: "{init_prompt}"

Enhance this instruction by adding a reasoning component.
Include phrases like "Let's think step by step" or "First, analyze...".

Write ONLY the enhanced instruction.

Enhanced instruction:"""

expert_generalist = """You are an expert at writing instructions for language models.

Task: {task_description}

Current instruction: "{init_prompt}"

Rewrite this instruction as if you were a world-class expert in this domain.
Make it:
- Clear and precise
- Generalizable (works across different cases)
- Professional but concise
- Focused on the core task

Do NOT mention specific examples. Write ONLY the improved instruction.

Improved instruction:"""

task_decomposition = """You are an expert at breaking down tasks into clear steps.

Task: {task_description}

Current instruction: "{init_prompt}"

Rewrite this instruction by clearly stating:
1. What to analyze
2. What to consider
3. What to output

Keep it concise. Write ONLY the improved instruction.

Improved instruction:"""

meta_task_description = """You are an expert in analyzing and understanding tasks deeply.

Original task description: {task_description}

Sample examples from this task:
{examples}

Based on your expertise and these examples, write a refined task description that:
1. Captures the CORE CHALLENGE and nuances of this specific task
2. Incorporates domain knowledge (key concepts, common pitfalls, edge cases)
3. Reflects the patterns you observe in the examples (without mentioning them explicitly)
4. Is concise but comprehensive (1-3 sentences maximum)

Think about: What makes this task difficult? What domain expertise is needed? What subtle patterns must be recognized?

Write ONLY the refined task description. No explanations, no examples, just the description.

Refined task description:""" 