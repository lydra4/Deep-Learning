FROM python:3.11.11-slim

ARG DEBIAN_FRONTEND="noninteractive"
ARG NON_ROOT_USER="horsesorhumans"
ARG NON_ROOT_UID="2222"
ARG NON_ROOT_GID="2222"
ARG HOME_DIR="/home/${NON_ROOT_USER}"
ARG REPO_DIR="."

# Create non-root user
RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER}

# Set environment
ENV PYTHONIOENCODING=utf8
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"
ENV PATH="${HOME_DIR}/.local/bin:${PATH}"

USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}

# Copy only the requirements file to leverage cache
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR}/requirements-gpu.txt ./requirements-gpu.txt

# Install pip requirements
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Copy the rest of the application code
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} .

# Set runtime environmen
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Default command
CMD ["python", "src/app.py"]