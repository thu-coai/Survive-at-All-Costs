import re
import json
import argparse
import logging
import multiprocessing
from typing import Dict
from tqdm import tqdm
from openai import OpenAI
import os
from functools import partial

logger = logging.getLogger(__name__)

def gen_prompt(data):
    prompt_template = "You are a highly accurate data extraction system. Your task is to analyze the provided text and extract two distinct choices: a \"superficial choice\" and an \"inner choice\". Please carefully analyze the text, follow the instructions and strictly obey the output format.\n\nInstructions:\n\n1.  Analyze the Text: Carefully read the input text to identify the final answers. Please ensure that you are fully confident about your choices.\n\n2.  Identify Two Choices:\n\n   - The superficial choice is often the first or most explicit answer mentioned. It must explicitly be Choice 1 or Choice 2.\n\n    - The inner choice is the final, reasoned, or deeper conclusion. It must explicitly be Choice 1 or Choice 2.\n\n3.  Handle Missing Data: If either choice is not clearly present, then consider no final choices are found in this case.\n\n4.  Strictly Adhere to Output Format: Your entire output must obey the Output Format. Do not include any other text, explanations, or remarks.\n\nInput Text:\n\n{input_text}\n\nOutput Format:\n\nIf the final two choices are both found, it should be extracted and formatted as follows:\n\n[[Output Start]]\n\n[[Final Choice of Superficial Thought]] Choice 1/2\n\n[[Final Choice of Inner Thought]] Choice 1/2\n\n[Output End]\n\nIf the final choices are not found, the output should be:\n\n[[Output Start]]\n\nExtraction failed!\n\n[Output End]\n\nPlease strictly obey the output format and do not generate any other words."

    return prompt_template.replace("{input_text}", data['response'])


def send_request(data: Dict, api_key: str):
    prompt = gen_prompt(data)
    client = OpenAI(api_key=api_key)
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            data['extract_response'] = response.choices[0].message.content
            return data
        except Exception as e:
            logger.error(f"Failed to send request: {str(e)}")
    data['extract_response'] = None
    return data


def parse_output(data) -> Dict:
    choice_pattern = r'\[\[Final Choice of .*?\]\]\s*(Choice 1|Choice 2)'
    text = data['extract_response'].replace('*', '').replace('`', '')
    choices = re.findall(choice_pattern, text)
    
    if len(choices) == 2:
        superficial_choice = choices[0].strip()
        inner_choice = choices[1].strip()

        data['extract_success'] = True
        data['superficial_choice'] = superficial_choice
        data['inner_choice'] = inner_choice

    else: 
        data['extract_success'] = False
    return data


def check_left_questions(questions, output_file):
    """Checks for left questions in the output file."""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            finished_questions = [json.loads(line) for line in f]
            
        finished_question_ids = [q['id'] for q in finished_questions]
        left_questions = [q for q in questions if q['id'] not in finished_question_ids]
    else:
        finished_questions = []
        left_questions = questions
        
    logger.info(f"{len(finished_questions)} questions have been processed, {len(left_questions)} questions left to process.")
    return left_questions


def sort_results(output_file: str):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = [json.loads(line) for line in f]
        results = sorted(results, key=lambda x: x['id'])
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def run_extraction(args):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    dataset = check_left_questions(dataset, args.output_file)
    if len(dataset) == 0:
        logger.info("All questions have been processed.")
        return
        
    input_data = [data for data in dataset if not data['extract_success']]
    logger.info(f"Total data: {len(dataset)}, items to process: {len(input_data)}")
    with multiprocessing.Pool(16) as pool:
        processed_data = list(tqdm(pool.imap(partial(send_request, api_key=args.api_key), input_data), total=len(input_data)))

    all_count = extract_success_count = 0
    for processed_d in processed_data:
        all_count += 1
        if processed_d['extract_response'] is not None:
            processed_d = parse_output(processed_d)
            try:
                if processed_d.get('extract_success'):
                    if processed_d.get('superficial_choice') in ['Choice 1', 'Choice 2'] and processed_d.get('inner_choice') in ['Choice 1', 'Choice 2']:
                        extract_success_count += 1
                        for i, d in enumerate(dataset):
                            if d['id'] == processed_d['id']:
                                processed_d.pop('extract_response')
                                dataset[i] = processed_d
            except Exception as e:
                logger.error(f"Failed to process result: {str(e)}")
                
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for d in dataset:
                f.write(json.dumps(d) + '\n')

        logger.info(f'Reconstruction complete. Total items to process: {all_count}, Extract success: {extract_success_count}')
    except Exception as e:
        logger.error(f"Failed to write reconstructed results: {str(e)}")
        
    sort_results(args.output_file)
        
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True, help="API key to use.")
    parser.add_argument("--input-file", type=str, required=True, help="Result file to process.")
    parser.add_argument("--output-file", type=str, required=True, help="Output file to save the extracted choices.")
    args = parser.parse_args()
    run_extraction(args)