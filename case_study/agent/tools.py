import subprocess
import os
from typing import Dict, Any, List

class ToolRegistry:
    """Tool Registry"""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register Default Tools"""
        self.register_tool("execute_command", self.execute_command)
        self.register_tool("list_directory", self.list_directory)
        self.register_tool("read_file", self.read_file)
        self.register_tool("write_file", self.write_file)
    
    def register_tool(self, name: str, func):
        """Register Tool"""
        self.tools[name] = func
    
    def get_tool_schemas(self) -> List[Dict]:
        """Get Tool Schemas"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute Command in Terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List Directory Content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the directory to list"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read File Content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path of the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write File Content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path of the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Tool"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' does not exist"}
        
        try:
            result = self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def execute_command(self, command: str) -> str:
        """Execute Command in Terminal"""
        check = input(f"Execute Command: {command}? (y/n/p)")
        if check == 'n':
            return "Command Execution Canceled"
        elif check == 'p':
            return "Command has been successfully executed"
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return f"Exit Code: {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Command Execution Timeout"
        except Exception as e:
            return f"Error executing command: {e}"
    
    def list_directory(self, path: str) -> str:
        """List Directory Content"""
        try:
            if not os.path.exists(path):
                return f"Path does not exist: {path}"
            
            items = os.listdir(path)
            result = []
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    result.append(f"[Directory] {item}")
                else:
                    result.append(f"[File] {item}")
            
            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    def read_file(self, file_path: str) -> str:
        """Read File Content"""
        try:
            if not os.path.exists(file_path):
                return f"File does not exist: {file_path}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {e}"
    
    def write_file(self, file_path: str, content: str) -> str:
        """Write File Content"""
        try:
            # Ensure Directory Exists
            if '/' in file_path:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File successfully written: {file_path}"
        except Exception as e:
            import traceback; traceback.print_exc()
            return f"Error writing file: {e}"