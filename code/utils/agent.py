import openai
import backoff
import time
import random
from openai import OpenAIError
from .openai_utils import OutOfQuotaException, AccessTerminatedException
from .openai_utils import num_tokens_from_string, model2max_context

support_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314']

class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float=0) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time

    @backoff.on_exception(backoff.expo, OpenAIError, max_tries=20)
    def query(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float) -> str:
        time.sleep(self.sleep_time)
        assert self.model_name in support_models, f"Not support {self.model_name}. Choices: {support_models}"
        
        try:
            client = openai.OpenAI(api_key=api_key)
            if self.model_name in support_models:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                gen = response.choices[0].message.content
            return gen
        except OpenAIError as e:
            error_message = str(e)
            if "You exceeded your current quota" in error_message:
                raise OutOfQuotaException(api_key)
            elif "Your access was terminated" in error_message:
                raise AccessTerminatedException(api_key)
            else:
                raise e

    def set_meta_prompt(self, meta_prompt: str):
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {self.name} -----\n{memory}\n")

    def ask(self, temperature: float=None):
        num_context_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])
        max_token = model2max_context[self.model_name] - num_context_token
        return self.query(self.memory_lst, max_token, api_key=self.openai_api_key, temperature=temperature if temperature else self.temperature)