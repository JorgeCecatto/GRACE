import requests
import uuid
import time
import random
from openai import OpenAI

import global_vars



class OpenAIModel():
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        base_model : bool,
        base_url: str = None,
        model_url_slug: str = None,
        provider_sort: str = None,
        max_tokens: int = None,
        provider_ignore: list = None,
        **kwargs):
        self.temperature = temperature
        self.model = model_name
        self.base_model = base_model
        # Preferencia de roteamento do OpenRouter (ex.: "price", "throughput",
        # "latency"). Enviada via extra_body em cada chamada; None = padrao do
        # provedor. Ver https://openrouter.ai/docs/features/provider-routing
        self.provider_sort = provider_sort
        # Lista de provedores do OpenRouter a ignorar no roteamento (campo
        # provider.ignore). None/vazio = nenhum provedor excluido.
        self.provider_ignore = provider_ignore
        # Teto de tokens da resposta (reasoning + conteudo final somados, para
        # modelos com reasoning). Evita chamadas presas gerando reasoning sem
        # nunca produzir a resposta final. Parametro padrao da API OpenAI,
        # None = sem teto explicito (usa o default do provedor).
        self.max_tokens = max_tokens

        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        # Comportamento de "model_url_slug":
        #   None -> derivado do model_name (sem o prefixo de organizacao, em
        #           minusculas e com "." trocado por "-"). Comportamento legado.
        #           Ex.: "Qwen/Qwen2.5-14B-Instruct-AWQ" -> "qwen2-5-14b-instruct-awq"
        #   ""   -> nenhum slug e anexado; a base_url e usada como esta. Usado
        #           por provedores como o OpenRouter, onde o modelo vai apenas
        #           no corpo da requisicao.
        #   str  -> usado tal como esta.
        if base_url is not None:
            slug = self._model_name_to_slug(model_name) if model_url_slug is None else model_url_slug
            resolved_base_url = base_url.rstrip("/")
            if slug:
                resolved_base_url = f"{resolved_base_url}/{slug}"
        else:
            resolved_base_url = "https://api.deepseek.com"

        self.client = OpenAI(api_key=api_key, base_url=resolved_base_url)

    @staticmethod
    def _model_name_to_slug(model_name: str) -> str:
        # Remove o prefixo de organizacao ("Qwen/...") e normaliza o nome para
        # o formato usado na url do servidor.
        name = model_name.split("/")[-1]
        return name.lower().replace(".", "-")


    def generate(self, input, temperature=None):
        assert isinstance(input, str)

        query = input.replace('"', '\\"')
        sleep_time = 20
        length_retry_sleep = 3
        max_retry = 5
        outputs = None
        resolved_temperature = self.temperature if temperature is None else temperature

        # Parametros nao-padrao da OpenAI (ex.: roteamento do OpenRouter) vao
        # via extra_body. Vazio para os demais provedores.
        extra_body = {}
        provider_prefs = {}
        if self.provider_sort:
            provider_prefs["sort"] = self.provider_sort
        if self.provider_ignore:
            provider_prefs["ignore"] = list(self.provider_ignore)
        if provider_prefs:
            extra_body["provider"] = provider_prefs

        create_kwargs = {}
        if self.max_tokens:
            create_kwargs["max_tokens"] = self.max_tokens

        next_sleep = sleep_time
        for i in range(int(max_retry + 1)):
            if i > 0:
                print(
                    f"Generation: retry {i}/{max_retry} after sleeping for {next_sleep:.0f} seconds."
                )
                time.sleep(next_sleep)
            next_sleep = sleep_time
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": query},
                    ],
                    stream=False,
                    temperature=resolved_temperature,
                    extra_body=extra_body,
                    **create_kwargs,
                )
                content = response.choices[0].message.content
                if content is None or not content.strip():
                    if response.choices[0].finish_reason == "length":
                        print(
                            "Generation: max_tokens atingido (finish_reason=length) "
                            "sem produzir resposta final."
                        )
                        next_sleep = length_retry_sleep
                    outputs = None
                    continue
                outputs = content.strip()

            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
            if outputs:
                break

        if self.base_model :
            global_vars.base_api_count +=1
            global_vars.base_input_token +=response.usage.prompt_tokens
            global_vars.base_output_token+=response.usage.completion_tokens
        else:
            global_vars.target_api_count +=1
            global_vars.target_input_token +=response.usage.prompt_tokens
            global_vars.target_output_token+=response.usage.completion_tokens

        return outputs

    
    def batch_forward_func(self, batch_prompts, temperature=None):
        outputs = [0]*len(batch_prompts)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {executor.submit(self.generate, batch_prompts[i], temperature): i for i in range(len(batch_prompts))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    outputs[index] = future.result()
                except Exception as e:
                    outputs[index] = ""
                    print(f"SYSTEM_ERROR: {str(e)}")

        return outputs