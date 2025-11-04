import requests
import time
from typing import List, Optional
import global_vars
import os

class OllamaModel:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        base_model: bool,
        base_url: str = None,
        **kwargs
    ):
        self.temperature = temperature
        self.model = model_name
        self.base_model = base_model

        if base_url is None:
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/api/chat"
        
        # Verifica se Ollama está acessível
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Não foi possível conectar ao Ollama em {self.base_url}: {e}")

    def generate(self, input: str) -> str:
        """
        Gera uma resposta a partir de um prompt.
        
        Args:
            input: prompt de entrada
            
        Returns:
            resposta do modelo como string
        """
        assert isinstance(input, str)
        
        sleep_time = 20
        max_retry = 5
        outputs = None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": input}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        for i in range(int(max_retry + 1)):
            if i > 0:
                print(f"Generation: retry {i}/{max_retry} after sleeping for {sleep_time:.0f} seconds.")
                time.sleep(sleep_time)
            
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=300  # 5 minutos timeout
                )
                response.raise_for_status()
                
                result = response.json()
                outputs = result.get('message', {}).get('content', '').strip()
                
                # Atualiza contadores globais
                # Nota: Ollama não retorna tokens como OpenAI
                # Você pode estimar ou deixar como 0
                if self.base_model:
                    global_vars.base_api_count += 1
                    global_vars.base_input_token += 0  # Ollama não retorna tokens
                    global_vars.base_output_token += 0
                else:
                    global_vars.target_api_count += 1
                    global_vars.target_input_token += 0
                    global_vars.target_output_token += 0
                
            except requests.exceptions.Timeout:
                print(f"Request timeout after 300s")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
            
            if outputs:
                break

        return outputs if outputs else ""

    def batch_forward_func(self, batch_prompts: List[str]) -> List[str]:
        """
        Processa um batch de prompts em paralelo.
        
        Args:
            batch_prompts: lista de prompts
            
        Returns:
            lista de respostas do modelo
        """
        outputs = [0] * len(batch_prompts)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Ollama local: use menos workers para não sobrecarregar
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_index = {
                executor.submit(self.generate, batch_prompts[i]): i 
                for i in range(len(batch_prompts))
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    outputs[index] = future.result()
                except Exception as e:
                    outputs[index] = ""
                    print(f"SYSTEM_ERROR: {str(e)}")

        return outputs