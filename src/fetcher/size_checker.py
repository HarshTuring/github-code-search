import requests

class RepoSizeChecker:
    """Utility for checking GitHub repository size before cloning."""
    
    def __init__(self, size_limit_mb=500):  # Default 500MB limit
        self.size_limit_mb = size_limit_mb
        
    def check_size(self, owner, repo):
        """Check if repository size is within acceptable limits.
        
        Returns:
            tuple: (is_within_limit, size_in_mb)
        """
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            raise ValueError(f"Failed to get repository data: {response.status_code}")
            
        data = response.json()
        # Size is in KB, convert to MB
        size_mb = data.get("size", 0) / 1024
        
        return (size_mb <= self.size_limit_mb, size_mb)