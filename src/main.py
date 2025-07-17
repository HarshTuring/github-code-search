import argparse
import sys
import os
from pathlib import Path
import logging
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from fetcher import RepositoryFetcher
from parser import RepositoryParser
from parser.models.chunk import Chunk

from embeddings import EmbeddingGenerator
from embeddings import VectorStore, VectorStoreManager
from query import QueryController
from src.processing.parallel_processor import ParallelRepositoryProcessor
import gc

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
                 size_limit_mb: int = 100,
                 embeddings_path: str = "./data/embeddings"):
        """
        Initialize the GitHub Code Analyzer.
        
        Args:
            base_repos_path: Directory to store downloaded repositories
            output_path: Directory to store parsed chunks
            size_limit_mb: Size limit for repositories in MB
            embeddings_path: Directory to store embeddings
        """
        self.base_repos_path = Path(base_repos_path)
        self.output_path = Path(output_path)
        self.size_limit_mb = size_limit_mb
        self.embeddings_path = Path(embeddings_path)
        
        # Ensure directories exist
        self.base_repos_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.fetcher = RepositoryFetcher(str(self.base_repos_path), size_limit_mb)
        self.vector_store_manager = VectorStoreManager(base_dir=str(self.embeddings_path))
        self.current_repo_path: Optional[Path] = None
        self.current_parser: Optional[RepositoryParser] = None
        self.current_chunks: List[Chunk] = []

    def process_repository_with_threading(self, repo_path, output_path, embeddings_path):
        """Process a repository with the parallel processor."""
        # Set up the parallel processor
        processor = ParallelRepositoryProcessor(
            repo_path=repo_path,
            min_workers=2,
            max_workers=os.cpu_count(),
            memory_threshold=80,
            cpu_threshold=90
        )
        
        # Define a progress callback
        def on_progress(status):
            print(f"Progress: {status['completion_percentage']:.2f}% - "
                  f"Processed: {status['processed_files']}/{status['total_files']} files")
        
        try:
            # Process the repository
            chunks = processor.process_repository(on_progress=on_progress)
            
            if not chunks:
                print("No chunks were generated.")
                return
                
            print(f"Generated {len(chunks)} chunks. Generating embeddings...")
            
            # Generate embeddings
            embedding_generator = EmbeddingGenerator(
                model_name="text-embedding-3-small",
                cache_dir=os.path.join(output_path, "embedding_cache")
            )
            
            # Process chunks in batches to avoid memory issues
            batch_size = 50
            all_embedded_chunks = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")
                
                embedded_chunks = embedding_generator.generate_embeddings(batch)
                all_embedded_chunks.extend(embedded_chunks)
                
                # Clear memory between batches
                del batch
                gc.collect()
            
            print(f"Generated embeddings for {len(all_embedded_chunks)} chunks")
            
            # Save the chunks to the vector store
            repo_name = Path(repo_path).name
            vector_store = VectorStore(repo_name, base_dir=embeddings_path)
            
            # Store in batches
            for i in range(0, len(all_embedded_chunks), batch_size):
                batch = all_embedded_chunks[i:i+batch_size]
                vector_store.add_chunks(batch)
            
            print(f"Repository processing complete. Chunks stored in vector database.")
            
        finally:
            # Clean up resources
            processor.shutdown()

    def store_embeddings(self, chunks: List[Chunk], repo_name: str) -> Dict[str, Any]:
        """
        Store embeddings for code chunks in a vector database.
        
        Args:
            chunks: List of code chunks with embeddings
            repo_name: Repository name for organizing storage
            
        Returns:
            Dictionary with storage statistics
        """
        chunks_with_embeddings = [c for c in chunks if c.metadata and 'embedding' in c.metadata]
        
        if not chunks_with_embeddings:
            logger.warning(f"No embeddings to store for {repo_name}")
            return {
                "success": False,
                "error": "No chunks with embeddings found",
                "repo_name": repo_name
            }
        
        try:
            # Get vector store for this repository
            vector_store = self.vector_store_manager.get_store(repo_name)
            
            # Add chunks to the vector store
            chunk_ids = vector_store.add_chunks(chunks_with_embeddings)
            
            # Get statistics
            stats = vector_store.get_stats()
            
            logger.info(f"Stored {len(chunk_ids)} embeddings in vector database for {repo_name}")
            
            return {
                "success": True,
                "chunks_stored": len(chunk_ids),
                "total_chunks": stats["chunk_count"],
                "repo_name": repo_name,
                "store_path": stats["path"]
            }
        except Exception as e:
            logger.error(f"Error storing embeddings for {repo_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "repo_name": repo_name
            }
        
    def analyze_github_repo(self, github_url: str, force_download: bool = False, 
                           verbose: bool = False, generate_embeddings: bool = False,
                           embedding_model: str = "text-embedding-3-large",
                           api_key: Optional[str] = None,
                           cache_embeddings: bool = True,
                           cache_dir: Optional[str] = None,
                           store_embeddings: bool = True) -> Dict[str, Any]:
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
            store_embeddings: Whether to store embeddings in a vector database
            
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
        
        try:
            self.process_repository_with_threading(
                repo_path=str(self.current_repo_path),
                output_path=str(self.output_path),
                embeddings_path=str(self.embeddings_path)
            )
            repo_name = self._get_repo_name_from_url(github_url)
            return {
                'success': True,
                'message': f'Successfully processed {repo_name} using parallel processing.',
                'repo_name': repo_name,
                'repo_path': str(self.current_repo_path)
            }
        except Exception as e:
            logger.error(f"Failed to process repository with threading: {e}")
            return {'success': False, 'error': f'Processing failed: {str(e)}'}

    def query_repository(self, repo_name: str, query_text: str, 
                     filters: Optional[Dict] = None, top_k: int = 5,
                     model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Query a repository that has been processed and stored in the vector database.
        
        Args:
            repo_name: Name of the repository to query
            query_text: Natural language query about the codebase
            filters: Optional filters to narrow search (language, chunk_type, etc.)
            top_k: Number of results to retrieve
            model: LLM model to use for response generation
            
        Returns:
            Dict with response data and source information
        """
        logger.info(f"Processing query for repository '{repo_name}': {query_text}")
        
        # Get the vector store for this repository
        try:
            vector_store = self.vector_store_manager.get_store(repo_name)
        except Exception as e:
            logger.error(f"Failed to get vector store for {repo_name}: {e}")
            return {
                "success": False,
                "error": f"Repository not found or no embeddings available: {str(e)}"
            }
        
        # Initialize components for the query pipeline
        try:
            # Create embedding generator
            # Load environment variables from .env file
            load_dotenv()
            
            # Get API key from environment variables
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to the .env file.")
                
            embedding_generator = EmbeddingGenerator(api_key=api_key)
            
            # Initialize query processor
            from query.query_processor import QueryProcessor
            query_processor = QueryProcessor(embedding_generator)
            
            # Initialize retrieval engine
            from query.retrieval_engine import RetrievalEngine
            retrieval_engine = RetrievalEngine(vector_store)
            
            # Initialize response generator
            from query.response_generator import ResponseGenerator
            
            response_generator = ResponseGenerator(model=model, api_key=api_key)
            
            # Process the query
            logger.info("Processing query and generating embeddings")
            query_data = query_processor.process_query(
                query_text=query_text,
                filters=filters
            )
            
            # Retrieve relevant chunks
            logger.info(f"Retrieving chunks (top_k={top_k})")
            retrieved_results = retrieval_engine.retrieve_with_context(
                query_data=query_data,
                top_k=top_k
            )

            # Print retrieved chunks for debugging
            print("\n--- Retrieved Chunks (Primary) ---")
            for i, chunk in enumerate(retrieved_results.get("primary_results", [])):
                print(f"[{i+1}] ID: {chunk.get('id')}")
                print(f"    File: {chunk.get('metadata', {}).get('file_path')}")
                print(f"    Type: {chunk.get('metadata', {}).get('chunk_type')}")
                print(f"    Language: {chunk.get('metadata', {}).get('language')}")
                print(f"    Similarity: {chunk.get('similarity')}")
                print(f"    Content (truncated): {chunk.get('content', '')[:200]}\n")
            print("--- Retrieved Chunks (Context) ---")
            for i, chunk in enumerate(retrieved_results.get("context_chunks", [])):
                print(f"[{i+1}] ID: {chunk.get('id')}")
                print(f"    File: {chunk.get('metadata', {}).get('file_path')}")
                print(f"    Type: {chunk.get('metadata', {}).get('chunk_type')}")
                print(f"    Language: {chunk.get('metadata', {}).get('language')}")
                print(f"    Content (truncated): {chunk.get('content', '')[:200]}\n")

            # Generate response
            logger.info("Generating response")
            response_data = response_generator.generate_response(
                query=query_text,
                retrieved_results=retrieved_results
            )

            # Add query success info and return
            response_data["success"] = True
            response_data["repository"] = repo_name

            return response_data
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "repository": repo_name
            }

    def interactive_query_mode(self, repo_name: str, model: str = "gpt-4o-mini"):
        """
        Start an interactive query session for a repository.
        
        Args:
            repo_name: Name of the repository to query
            model: LLM model to use for response generation
        """
        logger.info(f"Starting interactive query mode for repository '{repo_name}'")
        
        # Check if the repository exists in the vector store
        try:
            vector_store = self.vector_store_manager.get_store(repo_name)
            stats = vector_store.get_stats()
            logger.info(f"Found repository with {stats['chunk_count']} chunks")
        except Exception as e:
            logger.error(f"Failed to get vector store for {repo_name}: {e}")
            print(f"Error: Repository '{repo_name}' not found or no embeddings available.")
            return False
        
        print("\n" + "="*50)
        print(f"Interactive Query Mode: {repo_name}")
        print("Ask questions about the codebase or type 'exit' to quit")
        print("="*50 + "\n")
        
        while True:
            query = input("\nYour question: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            # Process the query
            response = self.query_repository(repo_name, query, model=model)
            
            if not response.get("success", False):
                print(f"\nError: {response.get('error', 'Unknown error')}")
                continue
            
            print("\n" + "-"*50)
            print("RESPONSE:")
            print(response["response"])
            print("\nSOURCES:")
            for source in response.get("sources", []):
                print(f"- {source['path']} ({source.get('chunk_type', 'code')})")
            print("-"*50)
            
            # Option to filter next query
            filter_choice = input("\nApply filters for next query? (y/n): ")
            if filter_choice.lower() == 'y':
                filters = {}
                
                language = input("Filter by language (press enter to skip): ")
                if language:
                    filters["language"] = language
                    
                chunk_type = input("Filter by chunk type (class/function/module, press enter to skip): ")
                if chunk_type:
                    filters["chunk_type"] = chunk_type
                    
                path_filter = input("Filter by file path (press enter to skip): ")
                if path_filter:
                    filters["path"] = path_filter
                    
                # Use filters for next query
                print(f"\nFilters applied: {filters}")
            
        print("\nExiting interactive mode.")
        return True

    def list_available_repositories(self):
        """
        List repositories available for querying (those with stored embeddings).
        
        Returns:
            List of repository names
        """
        try:
            return self.vector_store_manager.list_repositories()
        except Exception as e:
            logger.error(f"Error listing repositories: {e}")
            return []
    
    def _get_repo_name_from_url(self, github_url: str) -> str:
        """
        Extract a repository name from a GitHub URL for use as a collection name.
        
        Args:
            github_url: GitHub repository URL
            
        Returns:
            Repository name in format "username_reponame"
        """
        # Clean the URL
        clean_url = github_url.rstrip('/')
        if clean_url.endswith('.git'):
            clean_url = clean_url[:-4]
        
        # Extract username and repo name
        parts = clean_url.split('/')
        if len(parts) >= 2:
            username = parts[-2]
            repo_name = parts[-1]
            return f"{username}_{repo_name}"
        
        # Fallback: use a hash of the URL
        import hashlib
        return hashlib.md5(github_url.encode()).hexdigest()
        
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
        python -m src.main --github_url https://github.com/username/repo-name
    
    Force download and use verbose output:
        python -m src.main --github_url https://github.com/username/repo-name --force --verbose
        
    Generate embeddings and store in vector database:
        python -m src.main --github_url https://github.com/username/repo-name --embeddings --store
    
    Query a repository:
        python -m src.main --query "How is the authentication system implemented?" --repo-name username_repo-name
        
    Start interactive query mode:
        python -m src.main --interactive --repo-name username_repo-name
        
    List available repositories for querying:
        python -m src.main --list-repos
        """
    )
    
    # Repository analysis options
    parser.add_argument("--github_url", help="GitHub repository URL to analyze")
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
        "--embeddings-dir", 
        default="./data/embeddings",
        help="Directory to store embeddings"
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
    parser.add_argument(
        "--store",
        action="store_true",
        help="Store embeddings in vector database for querying"
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Disable vector storage even when generating embeddings"
    )
    
    # Query options (new)
    query_group = parser.add_argument_group('Repository Querying')
    query_group.add_argument(
        "--query", 
        help="Query to ask about the repository"
    )
    query_group.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive query mode"
    )
    query_group.add_argument(
        "--repo-name",
        help="Name of the repository to query"
    )
    query_group.add_argument(
        "--list-repos",
        action="store_true",
        help="List repositories available for querying"
    )
    query_group.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve for each query"
    )
    query_group.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model to use for generating responses"
    )
    
    # Query filters
    filter_group = parser.add_argument_group('Query Filters')
    filter_group.add_argument(
        "--filter-language",
        help="Filter by programming language"
    )
    filter_group.add_argument(
        "--filter-type",
        help="Filter by chunk type (function, class, etc.)"
    )
    filter_group.add_argument(
        "--filter-path",
        help="Filter by file path"
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
            size_limit_mb=args.size_limit,
            embeddings_path=args.embeddings_dir
        )
        
        # Check if we're in query mode
        if args.query or args.interactive or args.list_repos:
            # List repositories mode
            if args.list_repos:
                repos = analyzer.list_available_repositories()
                print("\nAvailable repositories for querying:")
                if repos:
                    for repo in repos:
                        print(f"- {repo}")
                else:
                    print("No repositories found with stored embeddings.")
                return 0
                
            # Make sure we have a repository name for querying
            if not args.repo_name and (args.query or args.interactive):
                logger.error("Repository name (--repo-name) is required for querying")
                return 1
            
            # Setup filters
            filters = {}
            if args.filter_language:
                filters["language"] = args.filter_language
            if args.filter_type:
                filters["chunk_type"] = args.filter_type
            if args.filter_path:
                filters["path"] = args.filter_path
                
            # Interactive mode
            if args.interactive:
                success = analyzer.interactive_query_mode(args.repo_name, model=args.llm_model)
                return 0 if success else 1
                
            # Single query mode
            if args.query:
                response = analyzer.query_repository(
                    args.repo_name,
                    args.query,
                    filters=filters if filters else None,
                    top_k=args.top_k,
                    model=args.llm_model
                )
                
                if response.get("success", False):
                    print("\n" + "="*50)
                    print("QUERY RESPONSE:")
                    print("="*50)
                    print(response["response"])
                    print("\nSOURCES:")
                    for source in response.get("sources", []):
                        print(f"- {source['path']} ({source.get('chunk_type', 'code')})")
                    return 0
                else:
                    logger.error(f"Query failed: {response.get('error', 'Unknown error')}")
                    return 1
                
        # Run repository analysis if github_url is provided
        elif args.github_url:
            # Run analysis
            result = analyzer.analyze_github_repo(
                args.github_url,
                force_download=args.force,
                verbose=args.verbose,
                generate_embeddings=args.embeddings,
                embedding_model=args.embedding_model,
                api_key=os.getenv("OPENAI_API_KEY"),
                cache_embeddings=not args.no_cache,
                cache_dir=args.cache_dir,
                store_embeddings=args.store and not args.no_store
            )
            
            if result['success']:
                logger.info("Analysis completed successfully")
                logger.info(f"Repository: {result['repository']}")
                logger.info(f"Downloaded to: {result['download_path']}")
                logger.info(f"Results saved to: {result['output_path']}")
                
                # Existing code to print statistics...
                # ...
                
                # If we stored embeddings, print the repository name for later querying
                if args.store and result['success'] and 'embeddings' in result:
                    embedding_info = result['embeddings']
                    if embedding_info.get('generated') and 'storage' in embedding_info:
                        storage_info = embedding_info['storage']
                        if storage_info['success']:
                            logger.info("\nTo query this repository later, use:")
                            logger.info(f"  --query \"your question\" --repo-name {result['name']}")
                            logger.info(f"  --interactive --repo-name {result['name']}")
                
                return 0
            else:
                logger.error(f"Analysis failed: {result['error']}")
                return 1
        else:
            parser.print_help()
            return 0
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())