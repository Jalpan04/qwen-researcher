import json
import zipfile
import random
import os

ZIP_FILE = "archive.zip"
JSON_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "arxiv_cs_2000.jsonl"
SAMPLE_SIZE = 2000

def process_data():
    if not os.path.exists(ZIP_FILE):
        print(f"Error: {ZIP_FILE} not found in the current directory.")
        return

    # Extract JSON file if it hasn't been extracted yet
    if not os.path.exists(JSON_FILE):
        print(f"Extracting {JSON_FILE} from {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extract(JSON_FILE)
    else:
        print(f"{JSON_FILE} already extracted. Skipping extraction.")
    
    print("Streaming JSON and filtering for 'cs.' categories...")
    cs_papers = []
    
    # We open and stream the file line by line to prevent massive memory usage
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                paper = json.loads(line)
                # Filter strictly for papers where category string includes "cs."
                if 'cs.' in paper.get('categories', ''):
                    cs_papers.append(paper)
            except json.JSONDecodeError:
                continue
    
    print(f"Found {len(cs_papers)} Computer Science papers.")
    print(f"Randomly sampling {SAMPLE_SIZE} papers...")
    
    if len(cs_papers) < SAMPLE_SIZE:
        print(f"Warning: Only found {len(cs_papers)} CS papers, which is less than requested.")
        sampled_papers = cs_papers
    else:
        sampled_papers = random.sample(cs_papers, SAMPLE_SIZE)
    
    print(f"Formatting to JSONL and saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for paper in sampled_papers:
            title = paper.get('title', '').strip()
            abstract = paper.get('abstract', '').strip()
            
            # Format using standard chat template structure
            formatted_item = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert post-doctoral computer science researcher. Provide accurate, highly technical summaries."
                    },
                    {
                        "role": "user", 
                        "content": f"Can you summarize the research paper titled {title}?"
                    },
                    {
                        "role": "assistant", 
                        "content": abstract
                    }
                ]
            }
            # Write out as a JSONL (one JSON object per line)
            out_f.write(json.dumps(formatted_item) + '\n')

    print(f"Data preparation complete! Check {OUTPUT_FILE}.")

if __name__ == "__main__":
    process_data()
