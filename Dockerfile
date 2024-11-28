# Use NVIDIA-compatible PyTorch base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://github.com/miike-ai/flux-fp8-setup.git flux-fp8-setup

# Change directory and run setup.py
WORKDIR /workspace/flux-fp8-setup

# Install dependencies and run the setup script
RUN pip install huggingface_hub[hf_transfer] gradio && \
    python setup.py

# Expose required ports
EXPOSE 7860
EXPOSE 8888

# Default command
CMD ["python", "setup.py"]
