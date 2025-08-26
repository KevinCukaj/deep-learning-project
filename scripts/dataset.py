import argparse
import yaml
import requests
import zipfile
import io
import os
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download and extract dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration with error handling
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"Error: Invalid YAML in config file '{args.config}'")
        sys.exit(1)

    # Validate config has required keys
    try:
        url = config['dataset']['url']
        root_dir = config['dataset']['root_dir']
    except KeyError as e:
        print(f"Error: Missing required key in config: {e}")
        sys.exit(1)

    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    
    print(f"Downloading from {url}...")
    
    # Download with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB chunks
        
        # Download with progress bar
        content = b''
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            content += data
        progress_bar.close()
        
        print("Download complete, extracting files...")
        
        # Extract with progress indication
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            file_list = z.namelist()
            for file in tqdm(file_list, desc="Extracting"):
                z.extract(file, path=root_dir)
                
        print(f"Extracted {len(file_list)} files to '{os.path.abspath(root_dir)}'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("Error: Downloaded content is not a valid zip file")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
