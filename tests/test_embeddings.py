import unittest
from unittest.mock import patch, MagicMock
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path to allow imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import GitHubCodeAnalyzer

class TestEmbeddingsGeneration(unittest.TestCase):
    @patch('main.RepositoryFetcher')
    @patch('main.RepositoryParser')
    @patch('main.GitHubCodeAnalyzer.save_chunks')
    @patch('main.GitHubCodeAnalyzer._get_output_dir_for_repo')
    @patch('main.Path')
    def test_embeddings_generation(self, MockPath, mock_get_output_dir, mock_save_chunks, MockRepositoryParser, MockRepositoryFetcher):
        # Arrange
        mock_fetcher_instance = MockRepositoryFetcher.return_value
        mock_fetcher_instance.fetch.return_value = '/fake/repo/path'
        
        mock_parser_instance = MockRepositoryParser.return_value
        mock_parser_instance.parse.return_value = MagicMock(file_count=10)
        mock_parser_instance.create_chunks.return_value = [
            MagicMock(language='python', chunk_type='function', content='def hello(): pass')
        ]

        mock_get_output_dir.return_value = '/fake/output/path'

        analyzer = GitHubCodeAnalyzer()
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.skipTest("OPENAI_API_KEY not found in .env file")

        # Act
        result = analyzer.analyze_github_repo(
            'https://github.com/HarshTuring/docklens',
            generate_embeddings=True,
            api_key=api_key
        )

        # Assert
        self.assertTrue(result['success'])
        self.assertTrue(result['embeddings']['generated'])

if __name__ == '__main__':
    unittest.main()
