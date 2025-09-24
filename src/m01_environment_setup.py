#!/usr/bin/env python3
"""
Module 01: Environment Setup & Requirements
Purpose: Create isolated environment with exact dependency versions for both training methodologies

Features:
- Virtual environment validation and creation
- CUDA/PyTorch compatibility checks
- HuggingFace authentication validation
- Database schema initialization
- Ollama installation preparation
- Comprehensive logging and error handling
"""

import os
import sys
import subprocess
import platform
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import importlib.util

# Import centralized database
sys.path.append(str(Path(__file__).parent))
from m00_centralized_db import CentralizedDB

class EnvironmentSetup:
    """Environment setup and validation for fine-tuning evaluation pipeline"""

    def __init__(self, project_name: str = "FntngEval", python_version: str = "3.11"):
        self.project_name = project_name
        self.python_version = python_version
        self.venv_name = f"venv_{project_name}"
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / self.venv_name
        self.requirements_path = self.project_root / "requirements.txt"
        self.db = CentralizedDB()

        # Setup logging
        log_dir = self.project_root / "outputs" / "m01_environment"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "environment_setup.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "machine": platform.machine()
        }

        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory_total_gb"] = round(memory.total / (1024**3), 2)
            info["memory_available_gb"] = round(memory.available / (1024**3), 2)
        except ImportError:
            info["memory_total_gb"] = None
            info["memory_available_gb"] = None

        return info

    def check_cuda_availability(self) -> Dict[str, Any]:
        """Check CUDA installation and GPU availability"""
        cuda_info = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_memory_gb": []
        }

        try:
            # Check NVIDIA-SMI
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, check=True)

            gpu_info = result.stdout.strip().split('\n')
            cuda_info["cuda_available"] = True
            cuda_info["gpu_count"] = len(gpu_info)

            for line in gpu_info:
                if line.strip():
                    name, memory = line.split(', ')
                    cuda_info["gpu_names"].append(name.strip())
                    cuda_info["gpu_memory_gb"].append(int(memory) / 1024)

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("NVIDIA-SMI not found or failed. CUDA may not be available.")

        # Check CUDA toolkit version
        try:
            result = subprocess.run(['nvcc', '--version'],
                                  capture_output=True, text=True, check=True)
            if "release" in result.stdout:
                version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
                cuda_version = version_line.split('release ')[1].split(',')[0]
                cuda_info["cuda_version"] = cuda_version
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("NVCC not found. CUDA development toolkit may not be installed.")

        return cuda_info

    def check_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment and virtual environment status"""
        env_info = {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "virtual_env": os.environ.get('VIRTUAL_ENV'),
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV'),
            "venv_exists": self.venv_path.exists(),
            "in_correct_venv": False,
            "pip_version": None
        }

        # Check if we're in correct virtual environment
        if env_info["virtual_env"]:
            env_info["in_correct_venv"] = self.venv_name in str(env_info["virtual_env"])

        # Get pip version
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                                  capture_output=True, text=True, check=True)
            env_info["pip_version"] = result.stdout.split()[1]
        except subprocess.CalledProcessError:
            self.logger.error("pip is not available")

        return env_info

    def check_system_python(self) -> Dict[str, Any]:
        """Check system Python (no installation - use Lambda.ai base image)"""
        python_info = {
            "python_available": False,
            "python_path": None,
            "python_version": None,
            "venv_supported": False,
            "error": None
        }

        try:
            # Use system Python (Lambda.ai base image)
            system_python = sys.executable
            result = subprocess.run([system_python, "--version"],
                                  capture_output=True, text=True, check=True)

            python_info["python_available"] = True
            python_info["python_path"] = system_python
            python_info["python_version"] = result.stdout.strip()

            # Check venv support
            result = subprocess.run([system_python, "-m", "venv", "--help"],
                                  capture_output=True, text=True)
            python_info["venv_supported"] = result.returncode == 0

            self.logger.info(f"Using Lambda.ai system Python: {python_info['python_version']}")
            return python_info

        except Exception as e:
            python_info["error"] = f"System Python check failed: {e}"
            self.logger.error(f"System Python check failed: {e}")
            return python_info

    def handle_existing_venv(self) -> bool:
        """Handle existing virtual environment with user confirmation"""
        if not self.venv_path.exists():
            return True  # No existing venv, proceed normally

        self.logger.warning(f"Existing virtual environment found at: {self.venv_path}")

        # Check if it's a Python 3.10 venv (needs upgrade)
        venv_python = None
        if platform.system() == "Windows":
            venv_python = self.venv_path / "Scripts" / "python.exe"
        else:
            venv_python = self.venv_path / "bin" / "python"

        needs_upgrade = False
        if venv_python.exists():
            try:
                result = subprocess.run([str(venv_python), "--version"],
                                      capture_output=True, text=True, check=True)
                if "Python 3.10" in result.stdout:
                    needs_upgrade = True
                    self.logger.warning("Existing venv uses Python 3.10 - upgrade to 3.11 recommended")
                elif "Python 3.11" in result.stdout:
                    self.logger.info("Existing venv already uses Python 3.11 - keeping it")
                    return True
            except subprocess.CalledProcessError:
                needs_upgrade = True  # Assume upgrade needed if can't determine version

        if needs_upgrade:
            print("\n" + "="*60)
            print("VIRTUAL ENVIRONMENT NOTICE")
            print("="*60)
            print(f"Existing venv path: {self.venv_path}")
            print("Current: Python 3.10 (Lambda.ai base image)")
            print("Status: Compatible with our requirements.txt")
            print("\nOptions:")
            print("  1. Keep existing venv (RECOMMENDED - no conflicts)")
            print("  2. Delete and recreate with same Python")
            print("  3. Exit and handle manually")

            while True:
                choice = input("\nChoose option (1/2/3): ").strip()
                if choice == "1":
                    self.logger.info("Keeping existing Python 3.10 venv - compatible with Lambda.ai")
                    return True
                elif choice == "2":
                    return self._delete_and_recreate_venv()
                elif choice == "3":
                    self.logger.info("User chose to exit and handle venv manually")
                    return False
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")

        return True

    def _delete_and_recreate_venv(self) -> bool:
        """Delete existing venv and prepare for recreation"""
        import shutil

        try:
            self.logger.info(f"Deleting existing virtual environment: {self.venv_path}")
            shutil.rmtree(self.venv_path)
            self.logger.info("Existing virtual environment deleted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete existing virtual environment: {e}")
            return False

    def create_virtual_environment(self) -> bool:
        """Create virtual environment using Lambda.ai system Python (no modifications)"""

        # Handle existing virtual environment
        if not self.handle_existing_venv():
            return False

        if self.venv_path.exists():
            self.logger.info(f"Using existing virtual environment at {self.venv_path}")
            return True

        # Check system Python first
        python_check = self.check_system_python()
        if not python_check["python_available"]:
            self.logger.error(f"System Python not available: {python_check.get('error', 'Unknown error')}")
            return False

        system_python = python_check["python_path"]
        self.logger.info(f"Using Lambda.ai system Python: {python_check['python_version']}")

        try:
            # Create venv with system Python (no installation or modification)
            self.logger.info(f"Creating virtual environment: {self.venv_name}")
            subprocess.run([system_python, '-m', 'venv', str(self.venv_path)],
                         check=True)
            self.logger.info("Virtual environment created successfully with Lambda.ai system Python")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            return False

    def install_requirements(self) -> Tuple[bool, Dict[str, Any]]:
        """Install requirements from requirements.txt"""
        if not self.requirements_path.exists():
            self.logger.error(f"Requirements file not found: {self.requirements_path}")
            return False, {}

        # Determine pip executable path
        if platform.system() == "Windows":
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_exe = self.venv_path / "bin" / "pip"

        if not pip_exe.exists():
            self.logger.error(f"pip not found in virtual environment: {pip_exe}")
            return False, {}

        install_info = {
            "requirements_file": str(self.requirements_path),
            "pip_executable": str(pip_exe),
            "install_success": False,
            "install_log": "",
            "error_log": ""
        }

        try:
            self.logger.info("Installing requirements...")

            # Upgrade pip first
            subprocess.run([str(pip_exe), 'install', '--upgrade', 'pip'],
                         check=True, capture_output=True, text=True)

            # Install requirements
            result = subprocess.run([str(pip_exe), 'install', '-r', str(self.requirements_path)],
                                  capture_output=True, text=True, timeout=1800)  # 30 minute timeout

            install_info["install_log"] = result.stdout
            install_info["error_log"] = result.stderr

            if result.returncode == 0:
                install_info["install_success"] = True
                self.logger.info("Requirements installed successfully")
            else:
                self.logger.error(f"Requirements installation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.logger.error("Requirements installation timed out after 30 minutes")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Requirements installation failed: {e}")
            install_info["error_log"] = str(e)

        return install_info["install_success"], install_info

    def install_flash_attention(self) -> Tuple[bool, Dict[str, Any]]:
        """Install FlashAttention2 with GPU-optimized compilation settings"""
        # Determine pip executable path
        if platform.system() == "Windows":
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_exe = self.venv_path / "bin" / "pip"

        flash_info = {
            "flash_attention_version": "2.8.3",
            "installation_method": "source_compilation",
            "gpu_optimized": True,
            "install_success": False,
            "install_log": "",
            "error_log": "",
            "compilation_time_minutes": 0
        }

        if not pip_exe.exists():
            self.logger.error(f"pip not found in virtual environment: {pip_exe}")
            return False, flash_info

        try:
            self.logger.info("Installing FlashAttention2 with GPU optimization...")
            self.logger.info("Note: This requires compilation and may take 10-20 minutes on first install")

            import time
            start_time = time.time()

            # Set environment variables for optimized compilation
            env = os.environ.copy()
            env['MAX_JOBS'] = '4'  # Limit parallel compilation jobs to prevent OOM
            env['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Specific to A10 GPU (compute capability 8.6)

            # Install FlashAttention2 with optimized settings
            result = subprocess.run([
                str(pip_exe), 'install',
                'flash-attn==2.8.3',
                '--no-build-isolation',  # Required for CUDA compilation
                '--verbose'
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=2400  # 40 minute timeout for compilation
            )

            compilation_time = (time.time() - start_time) / 60
            flash_info["compilation_time_minutes"] = round(compilation_time, 2)

            flash_info["install_log"] = result.stdout
            flash_info["error_log"] = result.stderr

            if result.returncode == 0:
                flash_info["install_success"] = True
                self.logger.info(f"FlashAttention2 installed successfully in {compilation_time:.1f} minutes")

                # Verify installation
                try:
                    import flash_attn
                    flash_info["verified_import"] = True
                    self.logger.info(f"FlashAttention2 version {flash_attn.__version__} verified")
                except ImportError:
                    flash_info["verified_import"] = False
                    self.logger.warning("FlashAttention2 installed but import failed")
            else:
                self.logger.error(f"FlashAttention2 installation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.logger.error("FlashAttention2 installation timed out after 40 minutes")
            flash_info["error_log"] = "Installation timeout - compilation took too long"
        except Exception as e:
            self.logger.error(f"FlashAttention2 installation failed: {e}")
            flash_info["error_log"] = str(e)

        return flash_info["install_success"], flash_info

    def validate_ml_packages(self) -> Dict[str, Any]:
        """Validate critical ML packages are properly installed"""

        # Get python executable from venv
        if platform.system() == "Windows":
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"

        validation_script = '''
import sys
import json
validation_results = {}

# Test PyTorch
try:
    import torch
    validation_results["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
except Exception as e:
    validation_results["torch"] = {"error": str(e)}

# Test Transformers
try:
    import transformers
    validation_results["transformers"] = {
        "version": transformers.__version__
    }
except Exception as e:
    validation_results["transformers"] = {"error": str(e)}

# Test PEFT
try:
    import peft
    validation_results["peft"] = {
        "version": peft.__version__
    }
except Exception as e:
    validation_results["peft"] = {"error": str(e)}

# Test BitsAndBytes
try:
    import bitsandbytes
    validation_results["bitsandbytes"] = {
        "version": bitsandbytes.__version__
    }
except Exception as e:
    validation_results["bitsandbytes"] = {"error": str(e)}

# Test other critical packages
packages = ["datasets", "accelerate", "trl", "numpy", "pandas", "sklearn"]
for pkg in packages:
    try:
        module = __import__(pkg)
        validation_results[pkg] = {
            "version": getattr(module, "__version__", "unknown")
        }
    except Exception as e:
        validation_results[pkg] = {"error": str(e)}

print(json.dumps(validation_results, indent=2))
'''

        try:
            result = subprocess.run([str(python_exe), '-c', validation_script],
                                  capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.error(f"Package validation failed: {e}")
            return {"validation_error": str(e)}

    def load_env_file(self):
        """Load environment variables from .env file"""
        env_file = self.project_root / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                self.logger.info("Environment variables loaded from .env file")
            except Exception as e:
                self.logger.warning(f"Failed to load .env file: {e}")

    def setup_huggingface_auth(self) -> Dict[str, Any]:
        """Setup and validate HuggingFace authentication"""
        # Load .env file first
        self.load_env_file()

        auth_info = {
            "token_found": False,
            "token_valid": False,
            "user_info": None,
            "login_successful": False,
            "error": None
        }

        try:
            # Check for HF token in environment
            token_names = ['HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN']
            hf_token = None

            for name in token_names:
                token = os.environ.get(name)
                if token and token != "your_huggingface_token_here":
                    hf_token = token
                    auth_info["token_found"] = True
                    self.logger.info(f"HuggingFace token found via {name}")
                    break

            if not hf_token:
                auth_info["error"] = "No HuggingFace token found. Please set HF_TOKEN in .env file"
                return auth_info

            # Validate token format
            if not (hf_token.startswith('hf_') and len(hf_token) >= 37):
                auth_info["error"] = "Invalid HuggingFace token format"
                return auth_info

            # Test token by logging in
            from huggingface_hub import login, whoami

            login(token=hf_token)
            auth_info["login_successful"] = True

            # Get user info
            user_info = whoami(token=hf_token)
            auth_info["token_valid"] = True
            auth_info["user_info"] = user_info

            self.logger.info(f"HuggingFace authentication successful! User: {user_info.get('name', 'Unknown')}")

        except ImportError:
            auth_info["error"] = "huggingface_hub not installed"
        except Exception as e:
            auth_info["error"] = str(e)

        return auth_info

    def check_huggingface_auth(self) -> Dict[str, Any]:
        """Check HuggingFace authentication status (legacy method)"""
        return self.setup_huggingface_auth()

    def install_ollama(self) -> Dict[str, Any]:
        """Install and setup Ollama"""
        ollama_info = {
            "ollama_installed": False,
            "version": None,
            "installation_attempted": False,
            "installation_successful": False,
            "service_running": False,
            "error": None
        }

        # Check if Ollama is already installed
        try:
            # Try both version commands
            try:
                result = subprocess.run(['ollama', '--version'],
                                      capture_output=True, text=True, check=True)
                version_output = result.stdout.strip()
            except subprocess.CalledProcessError:
                result = subprocess.run(['ollama', 'version'],
                                      capture_output=True, text=True, check=True)
                version_output = result.stdout.strip()

            ollama_info["ollama_installed"] = True
            ollama_info["version"] = version_output
            self.logger.info(f"Ollama already installed: {ollama_info['version']}")

            # Check if service is running
            try:
                result = subprocess.run(['ollama', 'list'],
                                      capture_output=True, text=True, check=True, timeout=10)
                ollama_info["service_running"] = True
                self.logger.info("Ollama service is running")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                self.logger.info("Ollama installed but service may not be running")

            return ollama_info

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.info("Ollama not found. Installing...")

        # Install Ollama
        ollama_info["installation_attempted"] = True
        system = platform.system().lower()

        try:
            if system == "linux":
                self.logger.info("Installing Ollama on Linux...")
                # Download and install Ollama
                subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'],
                             stdout=subprocess.PIPE, check=True)
                result = subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'],
                                      stdout=subprocess.PIPE, text=True, check=True)
                # Execute the install script
                subprocess.run(['sh'], input=result.stdout, text=True, check=True)

            elif system == "darwin":  # macOS
                self.logger.info("Installing Ollama on macOS...")
                subprocess.run(['brew', 'install', 'ollama'], check=True)

            else:
                ollama_info["error"] = f"Automatic installation not supported on {system}. Please install manually."
                return ollama_info

            # Verify installation
            try:
                result = subprocess.run(['ollama', '--version'],
                                      capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                result = subprocess.run(['ollama', 'version'],
                                      capture_output=True, text=True, check=True)
            ollama_info["ollama_installed"] = True
            ollama_info["installation_successful"] = True
            ollama_info["version"] = result.stdout.strip()
            self.logger.info(f"Ollama installed successfully: {ollama_info['version']}")

            # Start Ollama service in background
            try:
                subprocess.Popen(['ollama', 'serve'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                self.logger.info("Ollama service started in background")

                # Wait a moment and check if service is responding
                import time
                time.sleep(3)

                result = subprocess.run(['ollama', 'list'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    ollama_info["service_running"] = True
                    self.logger.info("Ollama service is running and responsive")

            except Exception as e:
                self.logger.warning(f"Failed to start Ollama service: {e}")

        except subprocess.CalledProcessError as e:
            ollama_info["error"] = f"Installation failed: {e}"
            self.logger.error(f"Failed to install Ollama: {e}")

        except Exception as e:
            ollama_info["error"] = f"Unexpected error: {e}"
            self.logger.error(f"Unexpected error during Ollama installation: {e}")

        return ollama_info

    def prepare_ollama_installation(self) -> Dict[str, Any]:
        """Prepare for Ollama installation (legacy method)"""
        return self.install_ollama()

    def run_complete_setup(self) -> Dict[str, Any]:
        """Run complete environment setup process"""
        self.logger.info("Starting complete environment setup...")

        setup_results = {
            "system_info": self.get_system_info(),
            "cuda_info": self.check_cuda_availability(),
            "python_env": self.check_python_environment(),
            "venv_created": False,
            "requirements_installed": False,
            "package_validation": {},
            "hf_auth": {},
            "ollama_preparation": {},
            "database_initialized": False,
            "setup_success": False
        }

        # Create virtual environment
        setup_results["venv_created"] = self.create_virtual_environment()

        if not setup_results["venv_created"]:
            self.logger.error("Failed to create virtual environment")
            return setup_results

        # Install requirements
        success, install_info = self.install_requirements()
        setup_results["requirements_installed"] = success
        setup_results["install_info"] = install_info

        # Install FlashAttention2 with GPU optimization (only if requirements succeeded)
        if success:
            self.logger.info("Installing FlashAttention2 for GPU optimization...")
            flash_success, flash_info = self.install_flash_attention()
            setup_results["flash_attention_installed"] = flash_success
            setup_results["flash_attention_info"] = flash_info

        if success:
            # Validate packages
            setup_results["package_validation"] = self.validate_ml_packages()

        # Setup HuggingFace auth
        setup_results["hf_auth"] = self.setup_huggingface_auth()

        # Install Ollama
        setup_results["ollama_installation"] = self.install_ollama()

        # Initialize database
        try:
            # Database is already initialized in __init__, just test it
            exp_id = self.db.create_experiment("environment_setup_test", "setup")
            self.db.update_experiment_status(exp_id, "completed")
            setup_results["database_initialized"] = True
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")

        # Log to database
        self.log_results_to_db(setup_results)

        # Determine overall success
        setup_results["setup_success"] = (
            setup_results["venv_created"] and
            setup_results["requirements_installed"] and
            setup_results["database_initialized"]
        )

        if setup_results["setup_success"]:
            self.logger.info("Environment setup completed successfully!")
        else:
            self.logger.error("Environment setup completed with errors")

        return setup_results

    def log_results_to_db(self, results: Dict[str, Any]):
        """Log setup results to centralized database"""
        try:
            # Extract key information for database
            db_data = {
                "python_version": results["system_info"].get("python_version"),
                "cuda_version": results["cuda_info"].get("cuda_version"),
                "memory_gb": results["system_info"].get("memory_total_gb"),
                "setup_status": "success" if results.get("setup_success") else "failed"
            }

            # Add package versions if available
            validation = results.get("package_validation", {})
            if "torch" in validation and "version" in validation["torch"]:
                db_data["torch_version"] = validation["torch"]["version"]
            if "transformers" in validation and "version" in validation["transformers"]:
                db_data["transformers_version"] = validation["transformers"]["version"]
            if "peft" in validation and "version" in validation["peft"]:
                db_data["peft_version"] = validation["peft"]["version"]

            # GPU information
            cuda_info = results.get("cuda_info", {})
            if cuda_info.get("gpu_names"):
                db_data["gpu_info"] = json.dumps({
                    "names": cuda_info["gpu_names"],
                    "memory_gb": cuda_info["gpu_memory_gb"],
                    "count": cuda_info["gpu_count"]
                })

            # Error logging
            if not results.get("setup_success"):
                error_log = []
                if not results.get("venv_created"):
                    error_log.append("Virtual environment creation failed")
                if not results.get("requirements_installed"):
                    error_log.append("Requirements installation failed")
                if not results.get("database_initialized"):
                    error_log.append("Database initialization failed")
                db_data["error_log"] = "; ".join(error_log)

            self.db.log_module_data("m01_environment", db_data)
            self.logger.info("Results logged to database successfully")

        except Exception as e:
            self.logger.error(f"Failed to log results to database: {e}")

    def generate_activation_script(self) -> str:
        """Generate script to activate environment and run next module"""
        if platform.system() == "Windows":
            activate_cmd = f"{self.venv_path}/Scripts/activate.bat"
            python_cmd = f"{self.venv_path}/Scripts/python.exe"
        else:
            activate_cmd = f"source {self.venv_path}/bin/activate"
            python_cmd = f"{self.venv_path}/bin/python"

        script_content = f"""#!/bin/bash
# Environment activation script for {self.project_name}
# Generated by m01_environment_setup.py

echo "Activating virtual environment: {self.venv_name}"
{activate_cmd}

echo "Verifying environment..."
{python_cmd} -c "import torch; print(f'PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')"

echo "Environment ready!"
echo "Run next module with:"
echo "{python_cmd} src/m02_data_preparation.py"
"""

        script_path = self.project_root / "activate_env.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable on Unix systems
        if platform.system() != "Windows":
            os.chmod(script_path, 0o755)

        return str(script_path)

def main():
    """Main execution function"""
    env_setup = EnvironmentSetup()

    print("=" * 60)
    print("FINE-TUNING EVALUATION - ENVIRONMENT SETUP")
    print("=" * 60)

    results = env_setup.run_complete_setup()

    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)

    print(f"System: {results['system_info']['platform']}")
    print(f"Python: {results['system_info']['python_version']}")
    print(f"Memory: {results['system_info'].get('memory_total_gb', 'Unknown')} GB")

    cuda_info = results['cuda_info']
    if cuda_info['cuda_available']:
        print(f"CUDA: {cuda_info['cuda_version']} ({cuda_info['gpu_count']} GPUs)")
        for i, (name, memory) in enumerate(zip(cuda_info['gpu_names'], cuda_info['gpu_memory_gb'])):
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    else:
        print("CUDA: Not available")

    print(f"Virtual Environment: {'✓' if results['venv_created'] else '✗'}")
    print(f"Requirements: {'✓' if results['requirements_installed'] else '✗'}")

    # FlashAttention status
    if results.get('flash_attention_installed'):
        flash_info = results.get('flash_attention_info', {})
        compile_time = flash_info.get('compilation_time_minutes', 0)
        print(f"FlashAttention2: ✓ (Compiled in {compile_time:.1f}m, GPU optimized for A10)")
    elif results.get('flash_attention_info'):
        print("FlashAttention2: ✗ (Compilation failed)")
    else:
        print("FlashAttention2: - (Not attempted)")

    print(f"Database: {'✓' if results['database_initialized'] else '✗'}")

    hf_auth = results['hf_auth']
    if hf_auth.get('token_valid') and hf_auth.get('login_successful'):
        print(f"HuggingFace: ✓ (User: {hf_auth['user_info'].get('name', 'Unknown')})")
    else:
        error_msg = hf_auth.get('error', 'Token not found or invalid')
        print(f"HuggingFace: ✗ ({error_msg})")

    ollama_info = results['ollama_installation']
    if ollama_info['ollama_installed']:
        service_status = "Running" if ollama_info.get('service_running') else "Installed"
        print(f"Ollama: ✓ ({service_status} - {ollama_info['version']})")
    else:
        error_msg = ollama_info.get('error', 'Not installed')
        print(f"Ollama: ✗ ({error_msg})")

    print("=" * 60)

    if results['setup_success']:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")

        # Generate activation script
        script_path = env_setup.generate_activation_script()
        print(f"📝 Activation script generated: {script_path}")
        print("\nNext steps:")
        print(f"1. Run: source {script_path}")
        print("2. Run: python src/m02_data_preparation.py")

    else:
        print("❌ SETUP COMPLETED WITH ERRORS")
        print("Please check the logs and resolve issues before proceeding.")

    return results

if __name__ == "__main__":
    main()