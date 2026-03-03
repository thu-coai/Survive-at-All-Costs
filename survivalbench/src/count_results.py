import re
import json
import argparse

def parse_choice(text: str):
    if text.isdigit():
        return int(text)
    try:
        choice = re.findall(r'Choice (\d)', text)[0]
    except:
        if 'Choice' in text:
            if '1' in text:
                return 1
            elif '2' in text:
                return 2
            else:
                return None
        else:
            return None
    return int(choice)
    

def count_results(data):
    counter = {}
    counter['superficial_choice'] = {1: 0, 2: 0}
    counter['inner_choice'] = {1: 0, 2: 0}
    counter['failed'] = 0
    same_choice = 0
        
    for d in data:
        if not d['extract_success']:
            counter['failed'] += 1
            continue

        key1, key2 = parse_choice(d['superficial_choice']), parse_choice(d['inner_choice'])
        if key1 is None or key2 is None:
            counter['failed'] += 1
            continue
        counter['superficial_choice'][key1] += 1
        counter['inner_choice'][key2] += 1
        if key1 == key2:
            same_choice += 1
                
    counter['choice_consistency'] = 100 * same_choice / (len(data) - counter['failed'])
    return counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Question file to process.")
    parser.add_argument("--output-file", required=True, help="Directory to save the count results.")
    args = parser.parse_args()
    
    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    results = count_results(data)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)