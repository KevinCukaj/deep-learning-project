import argparse
import yaml
import requests
import zipfile
import io
import os

def main():
    parser = argparse.ArgumentParser(description="Download and extract dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    url = config['dataset']['url']
    # Make request to get content
    response = requests.get(url)
    response.raise_for_status()  # Raise error if download failed

    # Use ZipFile with BytesIO
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(path=config['dataset']['root_dir'])
        print(f"Extracted {len(z.namelist())} files to '{os.path.abspath(config['dataset']['root_dir'])}'")

# Example usage:
if __name__ == "__main__":
    main()
