import os
import re
import yaml
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from ..models.file_info import FileInfo
from ..models.chunk import Chunk
from .base_parser import BaseParser

class InfrastructureParser(BaseParser):
    """Parser for infrastructure and deployment configuration files."""
    
    # Configuration file types by extension and naming pattern
    CONFIG_TYPES = {
        "docker": [
            # Dockerfile patterns
            r"Dockerfile$", r"Dockerfile\.[a-zA-Z0-9_-]+$",
            # Docker Compose
            r"docker-compose\.ya?ml$", r"docker-compose\.[a-zA-Z0-9_-]+\.ya?ml$",
        ],
        "kubernetes": [
            r".*\.k8s\.ya?ml$", r"kubernetes/.*\.ya?ml$",
            r"manifests/.*\.ya?ml$", r"k8s/.*\.ya?ml$",
            r".*deployment\.ya?ml$", r".*service\.ya?ml$",
            r".*ingress\.ya?ml$", r".*configmap\.ya?ml$",
            r".*secret\.ya?ml$", r".*volume\.ya?ml$",
        ],
        "ci_cd": [
            # GitHub Actions
            r"\.github/workflows/.*\.ya?ml$",
            # GitLab CI
            r"\.gitlab-ci\.ya?ml$",
            # CircleCI
            r"\.circleci/config\.ya?ml$",
            # Travis CI
            r"\.travis\.ya?ml$",
            # Jenkins
            r"Jenkinsfile$",
        ],
        "terraform": [
            r".*\.tf$", r".*\.tfvars$",
        ],
        "cloud": [
            # AWS CloudFormation and SAM
            r".*\.cloudformation\.ya?ml$", r"template\.ya?ml$",
            r".*\.sam\.ya?ml$",
            # Azure ARM Templates
            r".*\.arm\.json$",
        ],
        "env": [
            r"\.env$", r"\.env\.[a-zA-Z0-9_-]+$",
            r".*\.properties$", r".*\.conf$",
        ],
        "shell": [
            r".*\.sh$", r"deploy\.sh$", r"setup\.sh$",
            r"start\.sh$", r"stop\.sh$", r"restart\.sh$",
        ]
    }
    
    # Maps file extensions to languages
    EXTENSION_MAP = {
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.tf': 'terraform',
        '.tfvars': 'terraform',
        '.sh': 'shell',
        '.properties': 'properties',
        '.conf': 'configuration',
    }
    
    def get_language(self) -> str:
        """Get the language this parser handles."""
        return "infrastructure"
    
    def parse_file(self, file_path: Path) -> FileInfo:
        """Parse an infrastructure or deployment file and extract information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine configuration type
            config_type = self._detect_config_type(file_path)
            
            # Determine language based on extension or content
            extension = file_path.suffix.lower()
            language = self.EXTENSION_MAP.get(extension, 'text')
            
            file_info = FileInfo(
                path=str(file_path),
                language=language,
                file_type=config_type or "infrastructure",
                size_bytes=os.path.getsize(file_path),
                is_binary=False
            )
            
            # Extract key information based on file type
            symbols = {}
            
            # For Docker files, extract stages, base images, exposed ports, etc.
            if config_type == "docker" and file_path.name.startswith("Dockerfile"):
                symbols = self._parse_dockerfile(content)
                
            # For Docker Compose, extract services
            elif config_type == "docker" and "docker-compose" in file_path.name:
                symbols = self._parse_docker_compose(content)
                
            # For Kubernetes, extract resources
            elif config_type == "kubernetes":
                symbols = self._parse_kubernetes(content)
                
            # For CI/CD, extract job/step information
            elif config_type == "ci_cd":
                symbols = self._parse_ci_cd(file_path, content)
                
            # Save the extracted symbols to the file info
            file_info.symbols = symbols
            
            return file_info
                
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            # Return with limited information
            return FileInfo(path=str(file_path), language="text", file_type="infrastructure", is_binary=False)
    
    def create_chunks(self, file_info: FileInfo) -> List[Chunk]:
        """Create chunks from a parsed infrastructure file."""
        chunks = []
        
        try:
            # Open the file to get content
            with open(file_info.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file type for better chunking strategy
            file_type = file_info.file_type or "infrastructure"
            
            if file_type == "docker" and Path(file_info.path).name.startswith("Dockerfile"):
                # For Dockerfiles, chunk by stages
                chunks.extend(self._chunk_dockerfile(file_info, content))
                
            elif file_type == "docker" and "docker-compose" in Path(file_info.path).name:
                # For Docker Compose, chunk by services
                chunks.extend(self._chunk_docker_compose(file_info, content))
                
            elif file_type == "kubernetes":
                # For Kubernetes, chunk by resource type
                chunks.extend(self._chunk_kubernetes(file_info, content))
                
            elif file_type == "ci_cd":
                # For CI/CD, chunk by pipeline stages/jobs
                chunks.extend(self._chunk_ci_cd(file_info, content))
                
            elif file_type == "terraform":
                # For Terraform, chunk by resource blocks
                chunks.extend(self._chunk_terraform(file_info, content))
                
            else:
                # For other files, create a single chunk
                chunk_id = f"infra_{uuid.uuid4().hex[:8]}_{file_type}_{Path(file_info.path).stem}"
                chunks.append(Chunk(
                    id=chunk_id,
                    content=content,
                    metadata={
                        "file_path": file_info.path,
                        "language": file_info.language,
                        "type": file_type,
                        "name": Path(file_info.path).name,
                        "description": f"{file_type.upper()} configuration"
                    }
                ))
            
            return chunks
            
        except Exception as e:
            print(f"Error creating chunks for {file_info.path}: {e}")
            # Create a fallback chunk with the whole file
            try:
                with open(file_info.path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                chunk_id = f"infra_{uuid.uuid4().hex[:8]}_fallback_{Path(file_info.path).stem}"
                return [Chunk(
                    id=chunk_id,
                    content=content,
                    metadata={
                        "file_path": file_info.path,
                        "language": file_info.language,
                        "type": "infrastructure",
                        "name": Path(file_info.path).name,
                        "description": "Infrastructure configuration (fallback)"
                    }
                )]
            except:
                # If even reading fails, return empty list
                return []
    
    def _detect_config_type(self, file_path: Path) -> Optional[str]:
        """Detect the type of configuration file based on its path and name."""
        file_str = str(file_path)
        
        for config_type, patterns in self.CONFIG_TYPES.items():
            for pattern in patterns:
                if re.search(pattern, file_str, re.IGNORECASE):
                    return config_type
                    
        return None
    
    def _parse_dockerfile(self, content: str) -> Dict[str, Any]:
        """Extract information from a Dockerfile."""
        symbols = {
            'type': 'dockerfile',
            'base_images': [],
            'stages': [],
            'exposed_ports': [],
            'env_vars': [],
            'commands': {}
        }
        
        # Extract FROM instructions (base images and stages)
        for match in re.finditer(r'^\s*FROM\s+([^\s]+)(?:\s+AS\s+([^\s]+))?', content, re.MULTILINE):
            base_image = match.group(1)
            stage = match.group(2)
            
            symbols['base_images'].append(base_image)
            if stage:
                symbols['stages'].append(stage)
        
        # Extract EXPOSE instructions (ports)
        for match in re.finditer(r'^\s*EXPOSE\s+(.+?)(?:\s*#.*)?$', content, re.MULTILINE):
            ports = match.group(1).strip().split()
            symbols['exposed_ports'].extend(ports)
        
        # Extract ENV instructions (environment variables)
        for match in re.finditer(r'^\s*ENV\s+([^\s]+)(?:\s+|\=)(.+?)(?:\s*#.*)?$', content, re.MULTILINE):
            env_name = match.group(1)
            env_value = match.group(2).strip()
            symbols['env_vars'].append(f"{env_name}={env_value}")
        
        # Count different commands
        for cmd in ['RUN', 'COPY', 'ADD', 'WORKDIR', 'CMD', 'ENTRYPOINT']:
            symbols['commands'][cmd] = len(re.findall(fr'^\s*{cmd}\s+', content, re.MULTILINE))
        
        return symbols
    
    def _parse_docker_compose(self, content: str) -> Dict[str, Any]:
        """Extract information from a docker-compose file."""
        symbols = {
            'type': 'docker-compose',
            'services': [],
            'networks': [],
            'volumes': []
        }
        
        try:
            # Try to parse as YAML
            compose_data = yaml.safe_load(content)
            
            if isinstance(compose_data, dict):
                # Extract services
                if 'services' in compose_data and isinstance(compose_data['services'], dict):
                    for service_name, service_config in compose_data['services'].items():
                        service_info = {
                            'name': service_name,
                            'image': service_config.get('image', ''),
                            'build': bool(service_config.get('build')),
                            'ports': service_config.get('ports', []),
                            'depends_on': service_config.get('depends_on', []),
                        }
                        symbols['services'].append(service_info)
                
                # Extract networks
                if 'networks' in compose_data and isinstance(compose_data['networks'], dict):
                    symbols['networks'] = list(compose_data['networks'].keys())
                
                # Extract volumes
                if 'volumes' in compose_data and isinstance(compose_data['volumes'], dict):
                    symbols['volumes'] = list(compose_data['volumes'].keys())
        
        except Exception as e:
            print(f"Error parsing docker-compose as YAML: {e}")
            # Fallback to regex extraction for service names
            for match in re.finditer(r'^\s{2}([a-zA-Z0-9_-]+):\s*$', content, re.MULTILINE):
                service_name = match.group(1)
                if service_name not in ['networks', 'volumes', 'version']:
                    symbols['services'].append({'name': service_name})
        
        return symbols
    
    def _parse_kubernetes(self, content: str) -> Dict[str, Any]:
        """Extract information from Kubernetes YAML."""
        symbols = {
            'type': 'kubernetes',
            'resources': []
        }
        
        try:
            # Handle multi-document YAML files
            documents = list(yaml.safe_load_all(content))
            
            for doc in documents:
                if not isinstance(doc, dict):
                    continue
                    
                kind = doc.get('kind')
                name = doc.get('metadata', {}).get('name')
                namespace = doc.get('metadata', {}).get('namespace', 'default')
                
                if kind and name:
                    resource = {
                        'kind': kind,
                        'name': name,
                        'namespace': namespace
                    }
                    
                    # Extract specific details based on resource kind
                    if kind == 'Deployment':
                        spec = doc.get('spec', {})
                        containers = spec.get('template', {}).get('spec', {}).get('containers', [])
                        if containers:
                            resource['containers'] = [
                                {'name': c.get('name'), 'image': c.get('image')}
                                for c in containers if c.get('name')
                            ]
                    
                    symbols['resources'].append(resource)
        
        except Exception as e:
            print(f"Error parsing Kubernetes YAML: {e}")
        
        return symbols
    
    def _parse_ci_cd(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract information from CI/CD configuration."""
        symbols = {
            'type': 'ci_cd',
            'platform': 'unknown',
            'jobs': []
        }
        
        file_name = file_path.name
        file_str = str(file_path)
        
        # Determine CI/CD platform
        if '.github/workflows' in file_str:
            symbols['platform'] = 'github_actions'
        elif '.gitlab-ci.yml' in file_str:
            symbols['platform'] = 'gitlab_ci'
        elif '.circleci/config.yml' in file_str:
            symbols['platform'] = 'circle_ci'
        elif '.travis.yml' in file_str:
            symbols['platform'] = 'travis_ci'
        elif 'Jenkinsfile' in file_name:
            symbols['platform'] = 'jenkins'
        
        try:
            # GitHub Actions specific parsing
            if symbols['platform'] == 'github_actions':
                workflow = yaml.safe_load(content)
                if 'jobs' in workflow and isinstance(workflow['jobs'], dict):
                    for job_name, job_config in workflow['jobs'].items():
                        job_info = {
                            'name': job_name,
                            'steps': len(job_config.get('steps', [])),
                            'runs-on': job_config.get('runs-on', '')
                        }
                        symbols['jobs'].append(job_info)
            
            # GitLab CI specific parsing
            elif symbols['platform'] == 'gitlab_ci':
                gitlab_config = yaml.safe_load(content)
                for key, value in gitlab_config.items():
                    if isinstance(value, dict) and 'stage' in value:
                        job_info = {
                            'name': key,
                            'stage': value.get('stage', ''),
                            'script': bool(value.get('script'))
                        }
                        symbols['jobs'].append(job_info)
        
        except Exception as e:
            print(f"Error parsing CI/CD configuration: {e}")
        
        return symbols
    
    def _chunk_dockerfile(self, file_info: FileInfo, content: str) -> List[Chunk]:
        """Split a Dockerfile into chunks based on stages or logical sections."""
        chunks = []
        lines = content.split('\n')
        
        # First, try to split by stages
        stages = []  # List of (stage_name, start_line, end_line)
        
        # Find all FROM instructions which might define stages
        for i, line in enumerate(lines):
            match = re.match(r'^\s*FROM\s+([^\s]+)(?:\s+AS\s+([^\s]+))?', line)
            if match:
                stage_name = match.group(2) or match.group(1)
                stages.append((stage_name, i, None))
        
        # Set end lines
        for i in range(len(stages) - 1):
            stages[i] = (stages[i][0], stages[i][1], stages[i+1][1] - 1)
        
        # Last stage goes to the end of file
        if stages:
            stages[-1] = (stages[-1][0], stages[-1][1], len(lines) - 1)
        
        # Create chunks for each stage
        for stage_name, start_line, end_line in stages:
            stage_content = '\n'.join(lines[start_line:end_line+1])
            
            chunk_id = f"docker_{uuid.uuid4().hex[:8]}_stage_{stage_name}"
            chunk = Chunk(
                id=chunk_id,
                content=stage_content,
                metadata={
                    "file_path": file_info.path,
                    "language": "dockerfile",
                    "type": "docker_stage",
                    "name": f"Stage: {stage_name}",
                    "start_line": start_line + 1,  # Convert to 1-indexed
                    "end_line": end_line + 1,      # Convert to 1-indexed
                    "description": f"Dockerfile stage: {stage_name}"
                }
            )
            chunks.append(chunk)
        
        # If no stages found, create a single chunk
        if not chunks:
            chunk_id = f"docker_{uuid.uuid4().hex[:8]}_dockerfile_{Path(file_info.path).stem}"
            chunk = Chunk(
                id=chunk_id,
                content=content,
                metadata={
                    "file_path": file_info.path,
                    "language": "dockerfile",
                    "type": "dockerfile",
                    "name": "Dockerfile",
                    "description": "Dockerfile configuration"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_docker_compose(self, file_info: FileInfo, content: str) -> List[Chunk]:
        """Split a docker-compose file into chunks based on services."""
        chunks = []
        
        try:
            # Parse as YAML
            compose_data = yaml.safe_load(content)
            
            if not isinstance(compose_data, dict) or 'services' not in compose_data:
                # If we can't parse or no services found, return whole file as one chunk
                chunk_id = f"docker_{uuid.uuid4().hex[:8]}_compose_{Path(file_info.path).stem}"
                return [Chunk(
                    id=chunk_id,
                    content=content,
                    metadata={
                        "file_path": file_info.path,
                        "language": "yaml",
                        "type": "docker_compose",
                        "name": "Docker Compose",
                        "description": "Docker Compose configuration"
                    }
                )]
            
            # Create a chunk for each service
            for service_name, service_config in compose_data['services'].items():
                # Extract the service section as YAML
                service_yaml = yaml.dump({service_name: service_config}, default_flow_style=False)
                
                chunk_id = f"docker_{uuid.uuid4().hex[:8]}_service_{service_name}"
                chunk = Chunk(
                    id=chunk_id,
                    content=f"# Service: {service_name}\n{service_yaml}",
                    metadata={
                        "file_path": file_info.path,
                        "language": "yaml",
                        "type": "docker_service",
                        "name": f"Service: {service_name}",
                        "description": f"Docker Compose service: {service_name}"
                    }
                )
                chunks.append(chunk)
            
            # Create a chunk for networks and volumes if they exist
            other_sections = {}
            for section in ['networks', 'volumes']:
                if section in compose_data:
                    other_sections[section] = compose_data[section]
            
            if other_sections:
                other_yaml = yaml.dump(other_sections, default_flow_style=False)
                
                chunk_id = f"docker_{uuid.uuid4().hex[:8]}_infrastructure_{Path(file_info.path).stem}"
                chunk = Chunk(
                    id=chunk_id,
                    content=f"# Docker Compose Infrastructure\n{other_yaml}",
                    metadata={
                        "file_path": file_info.path,
                        "language": "yaml",
                        "type": "docker_infrastructure",
                        "name": "Docker Infrastructure",
                        "description": "Docker Compose networks and volumes"
                    }
                )
                chunks.append(chunk)
        
        except Exception as e:
            print(f"Error chunking docker-compose file: {e}")
            # Fallback to a single chunk
            chunk_id = f"docker_{uuid.uuid4().hex[:8]}_compose_fallback_{Path(file_info.path).stem}"
            chunk = Chunk(
                id=chunk_id,
                content=content,
                metadata={
                    "file_path": file_info.path,
                    "language": "yaml",
                    "type": "docker_compose",
                    "name": "Docker Compose (fallback)",
                    "description": "Docker Compose configuration"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_kubernetes(self, file_info: FileInfo, content: str) -> List[Chunk]:
        """Split Kubernetes YAML into chunks based on resource types."""
        chunks = []
        
        try:
            # Split multi-document YAML
            documents = list(yaml.safe_load_all(content))
            
            # Find document boundaries in the content
            doc_boundaries = []
            doc_start_pattern = re.compile(r'^---\s*$', re.MULTILINE)
            
            # Find all document start markers
            starts = [match.start() for match in doc_start_pattern.finditer(content)]
            
            # Handle case where first document doesn't start with ---
            if not starts or starts[0] > 0:
                starts.insert(0, 0)
            
            # Create boundaries (start, end) for each document
            for i in range(len(starts) - 1):
                doc_boundaries.append((starts[i], starts[i+1]))
            
            # Last document goes to the end
            if starts:
                doc_boundaries.append((starts[-1], len(content)))
            
            # Process each document
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    continue
                
                kind = doc.get('kind')
                name = doc.get('metadata', {}).get('name')
                namespace = doc.get('metadata', {}).get('namespace', 'default')
                
                if not kind or not name:
                    continue
                
                # Get document content
                if i < len(doc_boundaries):
                    start, end = doc_boundaries[i]
                    doc_content = content[start:end].strip()
                    if doc_content.startswith('---'):
                        doc_content = doc_content[3:].strip()
                else:
                    # Fallback: dump the document back to YAML
                    doc_content = yaml.dump(doc, default_flow_style=False)
                
                chunk_id = f"k8s_{uuid.uuid4().hex[:8]}_{kind.lower()}_{name}"
                chunk = Chunk(
                    id=chunk_id,
                    content=doc_content,
                    metadata={
                        "file_path": file_info.path,
                        "language": "yaml",
                        "type": f"kubernetes_{kind.lower()}",
                        "name": f"{kind}: {name}",
                        "namespace": namespace,
                        "description": f"Kubernetes {kind} resource: {name}"
                    }
                )
                chunks.append(chunk)
            
            # If no chunks were created, fallback to whole file
            if not chunks:
                raise Exception("No valid Kubernetes resources found")
        
        except Exception as e:
            print(f"Error chunking Kubernetes YAML: {e}")
            # Fallback to a single chunk
            chunk_id = f"k8s_{uuid.uuid4().hex[:8]}_fallback_{Path(file_info.path).stem}"
            chunk = Chunk(
                id=chunk_id,
                content=content,
                metadata={
                    "file_path": file_info.path,
                    "language": "yaml",
                    "type": "kubernetes",
                    "name": "Kubernetes Resource",
                    "description": "Kubernetes configuration"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_ci_cd(self, file_info: FileInfo, content: str) -> List[Chunk]:
        """Split CI/CD configuration into chunks based on jobs or stages."""
        chunks = []
        platform = file_info.symbols.get('platform', 'unknown')
        
        try:
            if platform == 'github_actions':
                # Parse GitHub Actions workflow
                workflow = yaml.safe_load(content)
                
                # First, create a chunk for workflow metadata (name, triggers)
                metadata = {k: v for k, v in workflow.items() if k != 'jobs'}
                if metadata:
                    metadata_yaml = yaml.dump(metadata, default_flow_style=False)
                    
                    chunk_id = f"cicd_{uuid.uuid4().hex[:8]}_workflow_metadata"
                    chunk = Chunk(
                        id=chunk_id,
                        content=f"# Workflow Metadata\n{metadata_yaml}",
                        metadata={
                            "file_path": file_info.path,
                            "language": "yaml",
                            "type": "github_actions_metadata",
                            "name": "Workflow Metadata",
                            "description": "GitHub Actions workflow triggers and metadata"
                        }
                    )
                    chunks.append(chunk)
                
                # Create a chunk for each job
                if 'jobs' in workflow and isinstance(workflow['jobs'], dict):
                    for job_name, job_config in workflow['jobs'].items():
                        job_yaml = yaml.dump({job_name: job_config}, default_flow_style=False)
                        
                        chunk_id = f"cicd_{uuid.uuid4().hex[:8]}_job_{job_name}"
                        chunk = Chunk(
                            id=chunk_id,
                            content=f"# Job: {job_name}\n{job_yaml}",
                            metadata={
                                "file_path": file_info.path,
                                "language": "yaml",
                                "type": "github_actions_job",
                                "name": f"Job: {job_name}",
                                "description": f"GitHub Actions job: {job_name}"
                            }
                        )
                        chunks.append(chunk)
            
            elif platform == 'gitlab_ci':
                # Parse GitLab CI config
                gitlab_config = yaml.safe_load(content)
                
                # Group jobs by stage
                stages = {}
                for key, value in gitlab_config.items():
                    if isinstance(value, dict) and 'stage' in value:
                        stage_name = value.get('stage', 'default')
                        if stage_name not in stages:
                            stages[stage_name] = []
                        stages[stage_name].append((key, value))
                
                # Create a chunk for each stage
                for stage_name, jobs in stages.items():
                    stage_jobs = {name: config for name, config in jobs}
                    stage_yaml = yaml.dump(stage_jobs, default_flow_style=False)
                    
                    chunk_id = f"cicd_{uuid.uuid4().hex[:8]}_stage_{stage_name}"
                    chunk = Chunk(
                        id=chunk_id,
                        content=f"# Stage: {stage_name}\n{stage_yaml}",
                        metadata={
                            "file_path": file_info.path,
                            "language": "yaml",
                            "type": "gitlab_ci_stage",
                            "name": f"Stage: {stage_name}",
                            "description": f"GitLab CI stage: {stage_name}"
                        }
                    )
                    chunks.append(chunk)
            
            # If no platform-specific chunking was done, return whole file
            if not chunks:
                chunk_id = f"cicd_{uuid.uuid4().hex[:8]}_{platform}_{Path(file_info.path).stem}"
                chunk = Chunk(
                    id=chunk_id,
                    content=content,
                    metadata={
                        "file_path": file_info.path,
                        "language": "yaml" if not platform == "jenkins" else "groovy",
                        "type": platform,
                        "name": f"{platform.replace('_', ' ').title()} Configuration",
                        "description": f"{platform.replace('_', ' ').title()} CI/CD configuration"
                    }
                )
                chunks.append(chunk)
        
        except Exception as e:
            print(f"Error chunking CI/CD file: {e}")
            # Fallback to a single chunk
            chunk_id = f"cicd_{uuid.uuid4().hex[:8]}_fallback_{Path(file_info.path).stem}"
            chunk = Chunk(
                id=chunk_id,
                content=content,
                metadata={
                    "file_path": file_info.path,
                    "language": file_info.language,
                    "type": "ci_cd",
                    "name": "CI/CD Configuration",
                    "description": "CI/CD configuration"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_terraform(self, file_info: FileInfo, content: str) -> List[Chunk]:
        """Split Terraform files into chunks based on resource blocks."""
        chunks = []
        
        # This is a simplified approach that uses regex to find resource blocks
        # A proper HCL parser would be better for production use
        
        try:
            # Add the full file as a single chunk for now (HCL parsing is complex)
            # In a production system, you'd use a proper HCL parser
            chunk_id = f"tf_{uuid.uuid4().hex[:8]}_config_{Path(file_info.path).stem}"
            chunk = Chunk(
                id=chunk_id,
                content=content,
                metadata={
                    "file_path": file_info.path,
                    "language": "terraform",
                    "type": "terraform",
                    "name": Path(file_info.path).name,
                    "description": "Terraform configuration"
                }
            )
            chunks.append(chunk)
            
            # TODO: Implement proper HCL parsing for better chunking
            
        except Exception as e:
            print(f"Error chunking Terraform file: {e}")
            # Fallback is already the whole file, so no need for another fallback
        
        return chunks