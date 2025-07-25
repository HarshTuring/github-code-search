import argparse
import sys
from pathlib import Path
from .repository_fetcher import RepositoryFetcher

def main():
    parser = argparse.ArgumentParser(description="Download GitHub repositories")
    parser.add_argument("url", help="GitHub repository URL")
    parser.add_argument("--output", "-o", help="Custom output directory name")
    parser.add_argument("--storage-path", default="./data/repos", 
                        help="Base storage path for repositories")
    parser.add_argument("--size-limit", type=int, default=100,
                        help="Size limit in MB (default: 100MB)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force download even if repository exceeds size limit")
    
    args = parser.parse_args()
    
    try:
        fetcher = RepositoryFetcher(args.storage_path, args.size_limit)
        repo_path = fetcher.fetch(args.url, args.output, force=args.force)
        
        if repo_path:
            print(f"Repository available at: {repo_path}")
            return 0
        else:
            print("Download cancelled by user.")
            return 2
            
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())