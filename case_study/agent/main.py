#!/usr/bin/env python3
"""
Main Program for Agent
Provides an interactive command line interface for interacting with the agent
"""

from agent import Agent
from loguru import logger
import argparse
import yaml
import json

def print_banner(agent):
    """Print Program Banner"""
    print("=" * 60)
    print(f"Welcome to {agent.agent_name}")
    print("This is an agent based on API, which can interact with the terminal through tools")
    print("Enter 'quit' or 'exit' to exit the program")
    print("Enter 'reset' to reset the conversation history")
    print("Enter 'help' to view help information")
    print("=" * 60)


def main(args):
    """Main Function"""
    # Load Config
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create Agent
    try:
        agent = Agent(config)
        logger.info(f"Agent {agent.agent_name} initialized, mode: {args.mode}")
        if args.mode == "interactive":
            print_banner(agent)
            
            while True:
                try:
                    # Get User Input
                    user_input = input(f"\n[{agent.agent_name}] Please Enter Your Message: ").strip()
                    
                    # Process Special Commands
                    if user_input.lower() in ['quit', 'exit']:
                        logger.info("Goodbye!")
                        break
                    elif user_input.lower() == 'reset':
                        agent.reset_conversation()
                        logger.info("Conversation History Reset")
                        continue
                    elif not user_input:
                        continue
                    
                    # Process User Message
                    logger.info(f"\n[User] {user_input}")
                    logger.info("\n[Agent] Processing...")
                    
                    response = agent.process_user_input(user_input)
                    logger.info(f"\n[Agent] {response}")
                    
                except KeyboardInterrupt:
                    logger.info("\n\nProgram Interrupted by User")
                    break
                except EOFError:
                    logger.info("\n\nProgram Ended")
                    break
                except Exception as e:
                    logger.error(f"\nError Occurred: {e}")
                    continue
                
            agent.save_conversation(save_conversation_file=args.save_conversation_file)
        
        elif args.mode == "autonomous":
            with open(agent.system_prompt_file, 'r') as f:
                prompt_file = json.load(f)
            for idx, dialogue_scenario in enumerate(prompt_file['dialogue_scenario']):
                user_prompt = dialogue_scenario['user_prompt']
                response = agent.process_user_input(user_prompt)
                logger.info(f"Round {idx+1} completed")
            agent.save_conversation(save_conversation_file=args.save_conversation_file)

    except Exception as e:
        logger.error(f"Error Initializing Agent: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Program")
    parser.add_argument("--config_file", type=str, help="Config File Path", default="config/deepseek_financial.yaml")
    parser.add_argument("--mode", type=str, help="Mode", choices=["interactive", "autonomous"], default="interactive")
    parser.add_argument("--save_conversation_file", type=str, help="Save Conversation File", default=None)
    args = parser.parse_args()
    
    main(args)