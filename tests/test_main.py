import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import GitHubCodeAnalyzer

class TestGitHubCodeAnalyzer(unittest.TestCase):

    @patch('main.RepositoryFetcher')
    @patch('main.RepositoryParser')
    @patch('main.GitHubCodeAnalyzer.save_chunks')
    @patch('main.GitHubCodeAnalyzer._get_output_dir_for_repo')
    @patch('main.Path')
    def test_analyze_github_repo_success(self, MockPath, mock_get_output_dir, mock_save_chunks, MockRepositoryParser, MockRepositoryFetcher):
        # Arrange
        mock_fetcher_instance = MockRepositoryFetcher.return_value
        mock_fetcher_instance.fetch.return_value = '/fake/repo/path'

        mock_parser_instance = MockRepositoryParser.return_value
        mock_parser_instance.parse.return_value = MagicMock(file_count=10)
        mock_parser_instance.create_chunks.return_value = [MagicMock(language='python', chunk_type='function', content='def hello(): pass')]

        mock_get_output_dir.return_value = '/fake/output/path'

        analyzer = GitHubCodeAnalyzer()

        # Act
        result = analyzer.analyze_github_repo('https://github.com/HarshTuring/docklens')

        # Assert
        self.assertTrue(result['success'])
        self.assertEqual(result['repository'], 'https://github.com/HarshTuring/docklens')
        self.assertIn('statistics', result)
        self.assertEqual(result['statistics']['total_chunks'], 1)
        mock_save_chunks.assert_called_once_with('/fake/output/path')
    
    @patch('main.RepositoryFetcher')
    @patch('main.RepositoryParser')
    @patch('main.GitHubCodeAnalyzer.save_chunks')
    @patch('main.GitHubCodeAnalyzer._get_output_dir_for_repo')
    @patch('main.Path')
    def test_embeddings_generation(self, MockPath, mock_get_output_dir, mock_save_chunks, MockRepositoryParser, MockRepositoryFetcher):
        mock_fetcher_instance = MockRepositoryFetcher.return_value
        mock_fetcher_instance.fetch.return_value = '/fake/repo/path'
        
        mock_parser_instance = MockRepositoryParser.return_value
        mock_parser_instance.parse.return_value = MagicMock(file_count=10)
        mock_parser_instance.create_chunks.return_value = [MagicMock(language='python', chunk_type='function', content='def hello(): pass')]

        mock_get_output_dir.return_value = '/fake/output/path'

        analyzer = GitHubCodeAnalyzer()

        # Act
        result = analyzer.analyze_github_repo('https://github.com/HarshTuring/docklens', generate_embeddings=True, api_key="")

        # Assert
        self.assertTrue(result['success'])
        self.assertTrue(result['embeddings']['generated'])

    @patch('main.RepositoryFetcher')
    @patch('main.RepositoryParser')
    @patch('main.GitHubCodeAnalyzer.save_chunks')
    @patch('main.GitHubCodeAnalyzer._get_output_dir_for_repo')
    @patch('main.Path')
    def test_caching(self, MockPath, mock_get_output_dir, mock_save_chunks, MockRepositoryParser, MockRepositoryFetcher):
        mock_fetcher_instance = MockRepositoryFetcher.return_value
        mock_fetcher_instance.fetch.return_value = '/fake/repo/path'
        
        mock_parser_instance = MockRepositoryParser.return_value
        mock_parser_instance.parse.return_value = MagicMock(file_count=10)
        mock_parser_instance.create_chunks.return_value = [MagicMock(language='python', chunk_type='function', content='def hello(): pass')]

        mock_get_output_dir.return_value = '/fake/output/path'

        analyzer = GitHubCodeAnalyzer()

        # Act
        result = analyzer.analyze_github_repo('https://github.com/HarshTuring/docklens', generate_embeddings=True, api_key="")

        # Assert
        self.assertTrue(result['success'])
        self.assertTrue(result['embeddings']['generated'])
        self.assertTrue(result['embeddings']['cache']['entries'] > 0)
    

if __name__ == '__main__':
    unittest.main()
