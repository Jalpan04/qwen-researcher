import json
import random
import os

def prepare_data():
    """
    Data preparation script to convert raw arXiv metadata into ChatML format.
    Handles memory efficiency by streaming the source file line-by-line.
    """
    
    input_file = "arxiv-metadata-oai-snapshot.json"
    output_file = "arxiv_cs_2000.jsonl"
    
    # Check if input exists
    if not os.path.exists(input_file):
        print(f"Source file {input_file} not found. Please ensure it is in the directory.")
        return

    cs_papers = []
    print("Scanning arXiv dataset for Computer Science papers...")

    # We use a streaming approach (open/read) to avoid loading the whole JSON into memory.
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                paper = json.loads(line)
                # Filter for papers containing 'cs.' in categories
                if 'cs.' in paper.get('categories', ''):
                    cs_papers.append({
                        'title': paper.get('title', 'No Title'),
                        'abstract': paper.get('abstract', 'No Abstract').replace('\n', ' ').strip()
                    })
            except json.JSONDecodeError:
                continue

    # Select a diverse sample of 2,000 papers for instruction tuning
    if len(cs_papers) > 2000:
        print(f"Found {len(cs_papers)} CS papers. Sampling 2,000...")
        selected_papers = random.sample(cs_papers, 2000)
    else:
        print(f"Found {len(cs_papers)} CS papers. Using all available data.")
        selected_papers = cs_papers

    # Convert to ChatML/Instruction format for Qwen
    # Each entry consists of a system prompt, a user question, and the assistant response.
    print(f"Writing formatted data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in selected_papers:
            # Construct the conversation
            messages = [
                {
                    "role": "system", 
                    "content": "You are a professional computer science researcher. Provide academic, detailed information based on research abstracts."
                },
                {
                    "role": "user", 
                    "content": f"Summarize the research and key contributions of the paper titled: {paper['title']}"
                },
                {
                    "role": "assistant", 
                    "content": paper['abstract']
                }
            ]
            
            # Write as a JSONL line
            json_line = json.dumps({"messages": messages})
            f.write(json_line + '\n')

    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
