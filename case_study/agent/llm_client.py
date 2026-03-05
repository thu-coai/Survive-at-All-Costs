from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger

class LLMClient:
    """LLM API Client"""
    
    def __init__(self, config: dict):
        self.api_key = config.get('api_key', None)
        self.api_base_url = config.get('api_base_url', None)
        self.model = config['model_name']
        self.max_tokens = config.get('max_tokens', 8192)
        self.temperature = config.get('temperature', 0.6)
        self.max_retries = config.get('max_retries', 20)
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not set")
            
        if self.api_base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
    

    def chat_completion(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Send Chat Request to LLM API"""
        
        for i in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    tool_choice="auto" if tools else None
                )
                res = {'content': None, 'cot': None, 'tool_calls': None}
                if response.choices[0].message.content is not None:
                    res['content'] = response.choices[0].message.content.strip()
                if response.choices[0].message.tool_calls is not None:
                    res['tool_calls'] = response.choices[0].message.tool_calls
                if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning is not None:
                    res['cot'] = response.choices[0].message.reasoning.strip()
                elif hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content is not None:
                    res['cot'] = response.choices[0].message.reasoning_content.strip()
                return res
            except Exception as e:
                logger.error(f"Error processing question for {i + 1} times: {e}")
                continue

        logger.error(f"Failed to process question after {self.max_retries} times.")
        return {'content': None, 'cot': None, 'tool_calls': None}
