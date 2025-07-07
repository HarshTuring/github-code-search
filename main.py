from src.fetcher import RepositoryFetcher

fetcher = RepositoryFetcher("./data/repositories")
repo_path = fetcher.fetch("https://github.com/HarshTuring/docklens")
print(f"Repository cloned to: {repo_path}")