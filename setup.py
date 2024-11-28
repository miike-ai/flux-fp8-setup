import os
import subprocess
import sys
import time


def run_command(command, env=None, cwd=None):
    """
    Run a shell command and handle errors.
    """
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, env=env, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        print(f"Error: {e}")
        sys.exit(1)


def install_dependencies():
    """
    Install required dependencies including huggingface_hub[hf_transfer] and gradio.
    """
    print("Installing dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]"])
    run_command([sys.executable, "-m", "pip", "install", "gradio"])
    requirements_file = os.path.join("flux-fp8-api", "requirements.txt")
    if os.path.exists(requirements_file):
        run_command([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    else:
        print(f"Requirements file not found: {requirements_file}")


def download_with_huggingface_cli(repo_id, filename, dest_dir):
    """
    Download a file using the huggingface-cli download command with hf_transfer enabled.
    """
    print(f"Downloading {repo_id}/{filename} to {dest_dir} using huggingface-cli...")
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["HF_HUB_DISABLE_SSL_VERIFY"] = "1"  # Add SSL verification bypass
    run_command(
        ["huggingface-cli", "download", repo_id, filename, "--local-dir", dest_dir],
        env=env,
    )
    print(f"Downloaded {filename} successfully to {dest_dir}.")


def setup_files(file_map, dest_folder):
    """
    Download files specified in the file_map into the specified folder using huggingface-cli.
    """
    os.makedirs(dest_folder, exist_ok=True)

    for filename, (repo_id, filename_in_repo) in file_map.items():
        dest_path = os.path.join(dest_folder, filename)
        download_with_huggingface_cli(repo_id, filename_in_repo, dest_folder)


def setup_t5_files(dest_folder):
    """
    Download all required T5 encoder files except for .gitattributes and README.md.
    """
    print("Setting up T5 encoder files...")
    t5_files = {
        "model.safetensors": ("city96/t5-v1_1-xxl-encoder-bf16", "model.safetensors"),
        "config.json": ("city96/t5-v1_1-xxl-encoder-bf16", "config.json"),
        "special_tokens_map.json": ("city96/t5-v1_1-xxl-encoder-bf16", "special_tokens_map.json"),
        "spiece.model": ("city96/t5-v1_1-xxl-encoder-bf16", "spiece.model"),
        "tokenizer_config.json": ("city96/t5-v1_1-xxl-encoder-bf16", "tokenizer_config.json"),
    }
    setup_files(t5_files, dest_folder)


def setup_model_files(dest_folder):
    """
    Download necessary model files into the specified folder.
    """
    print("Setting up model files...")
    model_files = {
        "flux1-schnell-fp8-e4m3fn.safetensors": ("Kijai/flux-fp8", "flux1-schnell-fp8-e4m3fn.safetensors"),
    }
    setup_files(model_files, dest_folder)


def setup_encoder_files(dest_folder):
    """
    Download files for the encoder into the specified folder.
    """
    print("Setting up encoder files...")
    encoder_files = {
        "flux-vae-bf16.safetensors": ("Kijai/flux-fp8", "flux-vae-bf16.safetensors"),
    }
    setup_files(encoder_files, dest_folder)


def clone_flux_repository():
    """
    Clone the required flux GitHub repository if it doesn't already exist.
    """
    repo_url = "https://github.com/miike-ai/flux-fp8-api.git"
    repo_name = "flux-fp8-api"

    if not os.path.exists(repo_name):
        print(f"Cloning repository from {repo_url}...")
        run_command(["git", "clone", repo_url])
    else:
        print(f"Repository {repo_name} already exists. Skipping clone.")


def run_main_script(repo_folder):
    """
    Change to the repo_folder and run the main script.
    """
    print(f"Changing directory to {repo_folder}...")
    os.chdir(repo_folder)
    print("Running main_gr.py...")
    run_command([sys.executable, "main_gr.py"])


if __name__ == "__main__":
    print("Starting setup process...")
    start_time = time.time()  # Start the timer

    try:
        # Clone flux repository and set up dependencies
        clone_flux_repository()

        # Set destination folder to the cloned repository
        flux_folder = "flux-fp8-api"

        # Install dependencies
        install_dependencies()

        # Download required files into the appropriate folders
        setup_model_files(flux_folder)
        setup_encoder_files(flux_folder)

        # Download T5 encoder files into the t5 folder within flux-fp8-api
        t5_folder = os.path.join(flux_folder, "t5")
        setup_t5_files(t5_folder)

        # Run main script from the flux-fp8-api folder
        run_main_script(flux_folder)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    end_time = time.time()  # End the timer
    duration = end_time - start_time
    print(f"\nProcess completed in {duration:.2f} seconds.")
