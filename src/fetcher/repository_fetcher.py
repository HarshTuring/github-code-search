import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import requests
from .size_checker import RepoSizeChecker

class RepositoryFetcher:
    """Handles downloading GitHub repositories to local storage."""
    
    def __init__(self, base_storage_path="./data/repos", size_limit_mb=100):
        self.base_storage_path = Path(base_storage_path)
        self.base_storage_path.mkdir(parents=True, exist_ok=True)
        self.size_checker = RepoSizeChecker(size_limit_mb)
        self.size_limit_mb = size_limit_mb
        
    def fetch(self, github_url, target_dir=None, force=False):
        """Fetch a GitHub repository and store it locally.
        
        Args:
            github_url: URL to GitHub repository
            target_dir: Optional custom directory name
            force: If True, skip size confirmation and download regardless of size
            
        Returns:
            Path to the cloned repository
            
        Raises:
            ValueError: For invalid URLs or non-existent repositories
            RuntimeError: For git operation failures
        """
        # Validate and parse GitHub URL
        repo_owner, repo_name = self._parse_github_url(github_url)
        
        # Determine target directory
        if not target_dir:
            target_dir = f"{repo_owner}_{repo_name}"
        
        repo_path = self.base_storage_path / target_dir
        
        # Check if repo already exists locally
        if repo_path.exists():
            print(f"Repository already exists at {repo_path}")
            return repo_path
        
        # Check repository existence on GitHub
        if not self._check_repo_exists(repo_owner, repo_name):
            raise ValueError(f"Repository {repo_owner}/{repo_name} does not exist or is not accessible")
        
        # Check repository size
        is_within_limit, size_mb = self.size_checker.check_size(repo_owner, repo_name)
        
        if not is_within_limit and not force:
            user_confirm = input(
                f"Repository size is {size_mb:.1f}MB, which exceeds the limit of {self.size_limit_mb}MB. "
                f"Do you want to continue? (y/n): "
            ).lower()
            
            if user_confirm != 'y' and user_confirm != 'yes':
                print("Download cancelled.")
                return None
        
        # Clone the repository
        clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"
        self._clone_repository(clone_url, repo_path)
        
        return repo_path
    
    def _parse_github_url(self, url):
        """Extract owner and repo name from GitHub URL."""
        # Support multiple GitHub URL formats
        github_patterns = [
            r"github\.com/([^/]+)/([^/]+)/?.*",
            r"github\.com:([^/]+)/([^/]+)\.git",
        ]
        
        parsed_url = urlparse(url)
        if not parsed_url.netloc.endswith("github.com"):
            raise ValueError("Not a GitHub URL")
            
        path = parsed_url.path.lstrip("/")
        
        for pattern in github_patterns:
            match = re.match(pattern, f"{parsed_url.netloc}/{path}")
            if match:
                owner, name = match.groups()
                # Remove .git suffix if present
                name = name.removesuffix(".git")
                return owner, name
        
        raise ValueError(f"Invalid GitHub URL format: {url}")
    
    def _check_repo_exists(self, owner, repo):
        """Verify that the repository exists and is accessible."""
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url)
        return response.status_code == 200
    
    def _clone_repository(self, clone_url, target_path):
        """Clone the repository to the target path."""
        try:
            print(f"Cloning from {clone_url} to {target_path}...")
            result = subprocess.run(
                ["git", "clone", "--progress", clone_url, str(target_path)],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=True
            )
            print(f"Repository cloned successfully to {target_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e.stderr}")
            # Clean up any partial clone
            if target_path.exists():
                import shutil
                shutil.rmtree(target_path)
            raise RuntimeError(f"Failed to clone repository: {e.stderr}")