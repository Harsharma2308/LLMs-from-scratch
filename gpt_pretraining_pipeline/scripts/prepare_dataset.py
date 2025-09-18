"""
Download multiple public domain texts from Project Gutenberg
for a medium-sized training dataset (~10MB)
"""

import os
import urllib.request
import tiktoken

# Public domain texts from Project Gutenberg
TEXTS = [
    {
        "name": "pride_and_prejudice.txt",
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "title": "Pride and Prejudice by Jane Austen"
    },
    {
        "name": "sherlock_holmes.txt", 
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "title": "The Adventures of Sherlock Holmes"
    },
    {
        "name": "alice_wonderland.txt",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "title": "Alice's Adventures in Wonderland"
    },
    {
        "name": "frankenstein.txt",
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
        "title": "Frankenstein by Mary Shelley"
    },
    {
        "name": "dracula.txt",
        "url": "https://www.gutenberg.org/files/345/345-0.txt",
        "title": "Dracula by Bram Stoker"
    }
]

def download_texts(output_dir="training_data"):
    """Download texts and combine them into a single training file"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_text = []
    total_size = 0
    
    print("Downloading texts from Project Gutenberg...")
    
    for text_info in TEXTS:
        filename = os.path.join(output_dir, text_info["name"])
        
        if not os.path.exists(filename):
            print(f"\nDownloading {text_info['title']}...")
            try:
                with urllib.request.urlopen(text_info["url"]) as response:
                    content = response.read().decode('utf-8')
                    
                # Save individual file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print(f"  ✓ Downloaded {len(content):,} characters")
            except Exception as e:
                print(f"  ✗ Failed to download: {e}")
                continue
        else:
            print(f"\n✓ {text_info['title']} already exists")
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        
        all_text.append(content)
        total_size += len(content)
        
        # Stop if we have enough data (targeting ~10MB)
        if total_size > 10_000_000:  # 10MB
            print(f"\nReached target size of 10MB")
            break
    
    # Combine all texts
    combined_file = os.path.join(output_dir, "combined_training_data.txt")
    combined_text = "\n\n=== NEW BOOK ===\n\n".join(all_text)
    
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    # Calculate statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print(f"Total size: {len(combined_text):,} characters ({len(combined_text)/1_000_000:.1f} MB)")
    print(f"Number of books: {len(all_text)}")
    
    # Count tokens
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(combined_text)
    print(f"Total tokens: {len(tokens):,}")
    print(f"Compression ratio: {len(combined_text)/len(tokens):.2f} chars per token")
    
    print(f"\nCombined training data saved to: {combined_file}")
    
    return combined_file, len(combined_text), len(tokens)

def create_train_val_split(input_file, train_ratio=0.9):
    """Split the combined text into train and validation sets"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    split_idx = int(train_ratio * len(text))
    
    output_dir = os.path.dirname(input_file)
    
    # Save train split
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(text[:split_idx])
    
    # Save validation split  
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(text[split_idx:])
    
    print(f"\nTrain/Val split created:")
    print(f"  Train: {len(text[:split_idx]):,} chars → {train_file}")
    print(f"  Val: {len(text[split_idx:]):,} chars → {val_file}")
    
    return train_file, val_file

if __name__ == "__main__":
    # Download and combine texts
    combined_file, total_chars, total_tokens = download_texts()
    
    # Create train/val split
    if total_chars > 0:
        train_file, val_file = create_train_val_split(combined_file)
        
        print("\n✅ Dataset ready for training!")
        print(f"   Use train.txt and val.txt for your experiments")
