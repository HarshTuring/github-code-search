import argparse
import json
import sys
import os
from pathlib import Path
from .repository_parser import RepositoryParser

def main():
    parser = argparse.ArgumentParser(description="Parse a repository into code chunks")
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("--output", "-o", help="Output directory for chunks")
    
    args = parser.parse_args()
    
    try:
        # Initialize parser
        repo_parser = RepositoryParser(args.repo_path)
        
        # Parse repository
        repo_structure = repo_parser.parse()
        print(f"Found {repo_structure.file_count} files in repository")
        
        # Create chunks
        chunks = repo_parser.create_chunks()
        print(f"Created {len(chunks)} code chunks")
        
        # Save chunks if output directory specified
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save each chunk to a separate file
            for chunk in chunks:
                chunk_file = output_path / f"{chunk.id}.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {chunk.name} ({chunk.chunk_type})\n")
                    f.write(f"# File: {chunk.file_path}\n")
                    f.write(f"# Lines: {chunk.start_line}-{chunk.end_line}\n\n")
                    f.write(chunk.content)
                    
            # Save metadata
            metadata_file = output_path / "chunks_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                chunks_data = [
                    {
                        "id": chunk.id,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks
                ]
                json.dump(chunks_data, f, indent=2)
                
            print(f"Saved {len(chunks)} chunks to {output_path}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())