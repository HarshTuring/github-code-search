import argparse
import sys
import os
from pathlib import Path
import logging
import json
from typing import Optional, List, Dict, Any

from fetcher import RepositoryFetcher
from parser import RepositoryParser
from parser.models.chunk import Chunk

from embeddings import EmbeddingGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class GitHubCodeAnalyzer:
    """Main controller class that orchestrates the GitHub repository analysis workflow."""
    
    def __init__(self, base_repos_path: str = "./data/repos", 
                 output_path: str = "./data/parsed",
                 size_limit_mb: int = 100):
        """
        Initialize the GitHub Code Analyzer.
        
        Args:
            base_repos_path: Directory to store downloaded repositories
            output_path: Directory to store parsed chunks
            size_limit_mb: Size limit for repositories in MB
        """
        self.base_repos_path = Path(base_repos_path)
        self.output_path = Path(output_path)
        self.size_limit_mb = size_limit_mb
        
        # Ensure directories exist
        self.base_repos_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.fetcher = RepositoryFetcher(str(self.base_repos_path), size_limit_mb)
        self.current_repo_path: Optional[Path] = None
        self.current_parser: Optional[RepositoryParser] = None
        self.current_chunks: List[Chunk] = []
        
    def analyze_github_repo(self, github_url: str, force_download: bool = False, 
                           verbose: bool = False, generate_embeddings: bool = False,
                           embedding_model: str = "text-embedding-3-large",
                           api_key: Optional[str] = None,
                           cache_embeddings: bool = True,
                           cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a GitHub repository - download, parse, and chunk it.
        
        Args:
            github_url: GitHub repository URL
            force_download: Force download even if size exceeds limit
            verbose: Enable verbose logging
            generate_embeddings: Whether to generate embeddings for chunks
            embedding_model: Model to use for generating embeddings
            api_key: OpenAI API key (if None, will use environment variable)
            cache_embeddings: Whether to use caching for embeddings
            cache_dir: Directory for the embedding cache (None for default)
            
        Returns:
            Dict with analysis results and statistics
        """
        logger.info(f"Starting analysis of repository: {github_url}")
        
        # Step 1: Fetch the repository
        try:
            repo_path = self.fetcher.fetch(github_url, force=force_download)
            if not repo_path:
                return {'success': False, 'error': 'Repository download was cancelled'}
            
            self.current_repo_path = repo_path
            logger.info(f"Repository downloaded to: {repo_path}")
        except Exception as e:
            logger.error(f"Failed to download repository: {e}")
            return {'success': False, 'error': f'Download failed: {str(e)}'}
        
        # Step 2: Parse the repository
        try:
            self.current_parser = RepositoryParser(str(repo_path))
            repo_structure = self.current_parser.parse()
            logger.info(f"Repository structure parsed: {repo_structure.file_count} files found")
        except Exception as e:
            logger.error(f"Failed to parse repository structure: {e}")
            return {'success': False, 'error': f'Parsing failed: {str(e)}'}
        
        # Step 3: Generate code chunks
        try:
            self.current_chunks = self.current_parser.create_chunks(verbose=verbose)
            logger.info(f"Generated {len(self.current_chunks)} code chunks")
        except Exception as e:
            logger.error(f"Failed to generate code chunks: {e}")
            return {'success': False, 'error': f'Chunking failed: {str(e)}'}

        embedding_info = {'generated': False}
        if generate_embeddings and self.current_chunks:
            try:
                logger.info(f"Generating embeddings for {len(self.current_chunks)} chunks using {embedding_model}")
                
                # Set cache directory if not specified
                if cache_dir is None and cache_embeddings:
                    cache_dir = str(self.output_path / "embedding_cache")
                    
                embedding_generator = EmbeddingGenerator(
                    model_name=embedding_model,
                    api_key=api_key,
                    use_cache=cache_embeddings,
                    cache_dir=cache_dir
                )
                
                self.current_chunks = embedding_generator.generate_embeddings(
                    self.current_chunks,
                    show_progress=verbose
                )
                
                embedding_count = sum(1 for chunk in self.current_chunks 
                                    if chunk.metadata and 'embedding' in chunk.metadata)
                
                # Get cache statistics if available
                cache_stats = {}
                if cache_embeddings and embedding_generator.cache:
                    cache_stats = embedding_generator.cache.get_stats()
                
                embedding_info = {
                    'generated': True,
                    'model': embedding_model,
                    'dimension': embedding_generator.dimension,
                    'count': embedding_count,
                    'cache': cache_stats if cache_stats else {'enabled': cache_embeddings}
                }
                
                logger.info(f"Successfully generated {embedding_count} embeddings")
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                embedding_info = {
                    'generated': False,
                    'error': str(e)
                }
        
        # Step 4: Save results
        try:
            output_dir = self._get_output_dir_for_repo(github_url)
            self.save_chunks(output_dir)
            logger.info(f"Analysis results saved to: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            return {'success': False, 'error': f'Saving results failed: {str(e)}'}
        
        # Compile statistics
        stats = self._generate_statistics()
        
        # Add embedding info to the result
        result = {
            'success': True,
            'repository': github_url,
            'download_path': str(self.current_repo_path),
            'output_path': str(output_dir),
            'statistics': stats,
            'embeddings': embedding_info
        }
        
        return result
    
    def save_chunks(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save the current chunks to files.
        
        Args:
            output_dir: Directory to save chunks. If None, a new directory will be created.
            
        Returns:
            Path to the output directory
        """
        if not self.current_chunks:
            raise ValueError("No chunks available to save. Run analyze_github_repo first.")
            
        if output_dir is None:
            if self.current_repo_path is None:
                raise ValueError("No repository is currently loaded")
            output_dir = self._get_output_dir_for_repo()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each chunk to a separate file
        for chunk in self.current_chunks:
            chunk_file = output_dir / f"{chunk.id}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(f"# {chunk.name} ({chunk.chunk_type})\n")
                f.write(f"# File: {chunk.file_path}\n")
                if chunk.start_line and chunk.end_line:
                    f.write(f"# Lines: {chunk.start_line}-{chunk.end_line}\n")
                f.write("\n")
                f.write(chunk.content)
        
        # Save chunk metadata (including embeddings)
        metadata_file = output_dir / "chunks_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            chunks_data = [
                {
                    "id": chunk.id,
                    "metadata": chunk.metadata
                }
                for chunk in self.current_chunks
            ]
            json.dump(chunks_data, f, indent=2)
            
        # If we have embeddings, save them separately for more efficient access
        chunks_with_embeddings = [c for c in self.current_chunks 
                                if c.metadata and 'embedding' in c.metadata]
                                
        if chunks_with_embeddings:
            embeddings_dir = output_dir / "embeddings"
            embeddings_dir.mkdir(exist_ok=True)
            
            # Save embeddings in a separate file for more targeted access
            embeddings_file = embeddings_dir / "embeddings.json"
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                embeddings_data = {
                    chunk.id: {
                        "embedding": chunk.metadata['embedding'],
                        "model": chunk.metadata.get('embedding_model', 'unknown'),
                        "dimension": chunk.metadata.get('embedding_dimension', len(chunk.metadata['embedding']))
                    }
                    for chunk in chunks_with_embeddings
                }
                json.dump(embeddings_data, f)
        
        # Save statistics
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            stats = self._generate_statistics()
            json.dump(stats, f, indent=2)
        
        return output_dir
    
    def get_chunks_by_type(self, chunk_type: str) -> List[Chunk]:
        """
        Get chunks of a specific type.
        
        Args:
            chunk_type: Type of chunks to retrieve
            
        Returns:
            List of chunks matching the specified type
        """
        return [c for c in self.current_chunks if c.chunk_type == chunk_type]
    
    def get_chunks_by_language(self, language: str) -> List[Chunk]:
        """
        Get chunks of a specific language.
        
        Args:
            language: Language of chunks to retrieve
            
        Returns:
            List of chunks written in the specified language
        """
        return [c for c in self.current_chunks if c.language == language]
    
    def _get_output_dir_for_repo(self, github_url: Optional[str] = None) -> Path:
        """
        Generate an output directory path for the current repository.
        
        Args:
            github_url: Optional GitHub URL to use instead of current repo
            
        Returns:
            Path object for the output directory
        """
        if github_url:
            # Extract owner and repo name from URL
            parts = github_url.rstrip('/').split('/')
            repo_name = parts[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            owner = parts[-2]
            return self.output_path / f"{owner}_{repo_name}"
        elif self.current_repo_path:
            return self.output_path / self.current_repo_path.name
        else:
            raise ValueError("No repository is currently loaded and no GitHub URL provided")
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about the current chunks.
        
        Returns:
            Dictionary with statistics
        """
        # Existing statistics generation code
        if not self.current_chunks:
            return {}
        
        # Count by language
        language_counts = {}
        for chunk in self.current_chunks:
            lang = chunk.language or 'unknown'
            if lang not in language_counts:
                language_counts[lang] = 0
            language_counts[lang] += 1
        
        # Count by type
        type_counts = {}
        for chunk in self.current_chunks:
            chunk_type = chunk.chunk_type or 'unknown'
            if chunk_type not in type_counts:
                type_counts[chunk_type] = 0
            type_counts[chunk_type] += 1
        
        # Calculate total content size
        total_size = sum(len(chunk.content) for chunk in self.current_chunks)
        
        # Count chunks with embeddings
        embedding_count = sum(1 for chunk in self.current_chunks 
                            if chunk.metadata and 'embedding' in chunk.metadata)
        
        # Check if we have embedding metadata to include
        embedding_info = {}
        if embedding_count > 0:
            # Get the embedding model and dimension from the first chunk with an embedding
            for chunk in self.current_chunks:
                if chunk.metadata and 'embedding' in chunk.metadata:
                    embedding_info = {
                        'model': chunk.metadata.get('embedding_model', 'unknown'),
                        'dimension': chunk.metadata.get('embedding_dimension', len(chunk.metadata['embedding'])),
                        'count': embedding_count
                    }
                    break
        
        stats = {
            'total_chunks': len(self.current_chunks),
            'by_language': language_counts,
            'by_type': type_counts,
            'total_content_size': total_size,
        }
        
        # Add embedding info if available
        if embedding_info:
            stats['embeddings'] = embedding_info
            
        return stats


def main():
    """Main entry point for the command line tool."""
    parser = argparse.ArgumentParser(
        description="GitHub Repository Code Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    Analyze a repository:
        python -m src.main https://github.com/username/repo-name
    
    Force download and use verbose output:
        python -m src.main https://github.com/username/repo-name --force --verbose
        
    Specify output directory:
        python -m src.main https://github.com/username/repo-name --output ./my_analysis

    Generate embeddings with specific model and disable caching:
        python -m src.main https://github.com/username/repo-name --embeddings --embedding-model text-embedding-3-large --no-cache
            """
    )
    
    parser.add_argument("github_url", help="GitHub repository URL to analyze")
    parser.add_argument(
        "--repos-dir", 
        default="./data/repos",
        help="Directory to store downloaded repositories"
    )
    parser.add_argument(
        "--output", "-o", 
        default="./data/parsed",
        help="Directory to store analysis results"
    )
    parser.add_argument(
        "--force", "-f", 
        action="store_true",
        help="Force download even if repository exceeds size limit"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--size-limit", 
        type=int, 
        default=100,
        help="Repository size limit in MB (default: 100)"
    )
    parser.add_argument(
        "--embeddings", "-e", 
        action="store_true",
        help="Generate embeddings for code chunks"
    )
    parser.add_argument(
        "--embedding-model",
        choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        default="text-embedding-3-small",
        help="OpenAI model to use for embeddings"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache"
    )
    parser.add_argument(
        "--cache-dir",
        help="Specify a custom directory for the embedding cache"
    )
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = GitHubCodeAnalyzer(
            base_repos_path=args.repos_dir,
            output_path=args.output,
            size_limit_mb=args.size_limit
        )
        # Run analysis
        result = analyzer.analyze_github_repo(
            args.github_url,
            force_download=args.force,
            verbose=args.verbose,
            generate_embeddings=args.embeddings,
            embedding_model=args.embedding_model,
            api_key="",
            cache_embeddings=not args.no_cache,
            cache_dir=args.cache_dir
        )
        
        if result['success']:
            logger.info("Analysis completed successfully")
            logger.info(f"Repository: {result['repository']}")
            logger.info(f"Downloaded to: {result['download_path']}")
            logger.info(f"Results saved to: {result['output_path']}")
            
            # Print statistics
            stats = result['statistics']
            logger.info("\nStatistics:")
            logger.info(f"  Total chunks: {stats['total_chunks']}")
            
            logger.info("\n  Chunks by language:")
            for lang, count in stats['by_language'].items():
                logger.info(f"    {lang}: {count}")
                
            logger.info("\n  Chunks by type:")
            for chunk_type, count in stats['by_type'].items():
                logger.info(f"    {chunk_type}: {count}")

            result_embedding_info = result['embeddings']

            if 'cache' in result_embedding_info and isinstance(result_embedding_info['cache'], dict) and 'hit_rate' in result_embedding_info['cache']:
                cache_stats = result_embedding_info['cache']
                logger.info(f"    Cache: {cache_stats['entries']} entries")
                logger.info(f"    Cache hits: {cache_stats['hits']}")
                logger.info(f"    Cache misses: {cache_stats['misses']}")
                logger.info(f"    Cache hit rate: {cache_stats['hit_rate']:.1%}")
    
            
            # Print embedding info if available
            if 'embeddings' in stats:
                embedding_info = stats['embeddings']
                logger.info("\n  Embeddings:")
                logger.info(f"    Model: {embedding_info['model']}")
                logger.info(f"    Dimension: {embedding_info['dimension']}")
                logger.info(f"    Count: {embedding_info['count']}")
            elif args.embeddings:
                embedding_info = result.get('embeddings', {})
                if embedding_info.get('generated'):
                    logger.info("\n  Embeddings:")
                    logger.info(f"    Model: {embedding_info['model']}")
                    logger.info(f"    Dimension: {embedding_info['dimension']}")
                    logger.info(f"    Count: {embedding_info['count']}")
                else:
                    logger.warning("\n  Embedding generation failed:")
                    if 'error' in embedding_info:
                        logger.warning(f"    Error: {embedding_info['error']}")
                
            return 0
        else:
            logger.error(f"Analysis failed: {result['error']}")
            return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())