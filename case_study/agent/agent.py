import json
import os
from typing import List, Dict
from llm_client import LLMClient
from tools import ToolRegistry
from loguru import logger

class Agent:
    """Agent Core Class"""
    
    def __init__(self, config: dict):
        self.config = config
        self.llm_client = LLMClient(config)
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
        self.responses = []
        self.model_name = config.get['model_name']
        self.agent_name = config.get('agent_name', 'Assistant')
        self.max_rounds = config['max_rounds']
        self.system_prompt_file = config['system_prompt_file']
        self.save_conversation_file = config['save_conversation_file']
        self.system_prompt = None
        self.turn_counter = 0
        self.set_system_prompt()
            
            
    def set_system_prompt(self):
        logger.info(f"Setting system prompt.")
        if self.system_prompt_file:
            with open(self.system_prompt_file, 'r') as f:
                self.system_prompt = json.load(f)['system_prompt']
        else:
            self.system_prompt = f"""You are {self.agent_name}, a helpful assistant. """
        self.add_message("user", self.system_prompt + '\n\nIf you understand your role, please reply with a single word "Yes".')
        response = self.llm_client.chat_completion(
            messages=self.get_messages(),
            tools=None
        )
        if 'yes' not in response['content'].lower():
            logger.error(f"System prompt response: {response['content']}")
            raise ValueError("System prompt response is not 'Yes', please check the system prompt file.")
        
        logger.info(f"System prompt is set successfully.")
        self.add_message("assistant", (response['cot'] + '\n\n') if response['cot'] else '' + response['content'])
        if response['tool_calls']:
            response['tool_calls'] = str(response['tool_calls'])
        self.responses.append({'turn': f'{self.turn_counter}', 'response': response})
        self.turn_counter += 1
    
    
    def add_message(self, role: str, content: str):
        """Add Message to Conversation History"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get Full Message List (Including System Message)"""
        return self.conversation_history
    
    def process_user_input(self, user_input: str) -> str:
        """Process User Input"""
        # Add User Message
        self.add_message("user", user_input)
        
        # Get Tool Schemas
        tools = self.tool_registry.get_tool_schemas()
        
        # Call LLM
        cur_turn = 0
        for i in range(self.max_rounds):
            try:
                logger.info(f"Round {i+1} of {self.max_rounds} rounds.")
                
                response = self.llm_client.chat_completion(
                    messages=self.get_messages(),
                    tools=tools
                )
                
                # logger.debug(response)
                
                # Check if there are tool calls
                tool_calls = response['tool_calls']

                if not tool_calls and not response['content']:
                    continue
                
                if tool_calls:
                    # Execute Tool Calls
                    cur_turn += 1
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        # Execute Tool
                        result = self.tool_registry.execute_tool(tool_name, arguments)
                        
                        # Add Tool Call Result to Conversation History
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(result, ensure_ascii=False)
                        })
                    
                    # Add Tool Calls and Results to Conversation History
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response['cot'] if response['cot'] else None,
                        "tool_calls": tool_calls
                    })
                    
                    for result in tool_results:
                        self.conversation_history.append(result)
                        
                    response['tool_calls'] = str(tool_calls)
                    self.responses.append({'turn': f'{self.turn_counter}_{cur_turn}', 'response': response, 'tool_results': tool_results})
                    
                else:
                    # No Tool Calls, Return Content Directly
                    cur_turn += 1
                    assistant_message = response['content']
                    self.add_message("assistant", (response['cot'] + '\n\n') if response['cot'] else '' + assistant_message)
                    if response['tool_calls']:
                        response['tool_calls'] = str(response['tool_calls'])
                    self.responses.append({'turn': f'{self.turn_counter}_{cur_turn}', 'response': response})
                    self.turn_counter += 1
                    return assistant_message
                    
            except Exception as e:
                error_message = f"Error processing request: {e}"
                logger.error(error_message)
                continue
    
    
    def reset_conversation(self):
        """Reset Conversation History"""
        self.conversation_history = []
    
    
    def get_conversation_summary(self) -> str:
        """Get Conversation Summary"""
        return f"Conversation History contains {len(self.conversation_history)} messages" 
    
    
    def save_conversation(self, save_conversation_file: str = None):
        """Save Conversation to File"""
        if save_conversation_file:
            self.save_conversation_file = save_conversation_file
        logger.info(f"Saving conversation to {self.save_conversation_file}")
        os.makedirs(os.path.dirname(self.save_conversation_file), exist_ok=True)
        with open(self.save_conversation_file, 'w') as f:
            json.dump(self.responses, f, ensure_ascii=False, indent=4)