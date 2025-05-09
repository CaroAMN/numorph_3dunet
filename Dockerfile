FROM continuumio/miniconda3:latest

LABEL maintainer="Carolin Schwitalla <carolin.schwitalla@uni-tuebingen.de>"
LABEL description="NuMorph 3D-UNet for nuclei segmentation in light-sheet microscopy data"

# Fix the apt sources list issue
#RUN sed -i 's/stable/oldoldstable/g' /etc/apt/sources.list && \
 #   sed -i 's/stable-updates/oldoldstable-updates/g' /etc/apt/sources.list

# Install git which might be needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy environment file first (better for caching)
COPY numorphunet.yml /app/

# Create conda environment
RUN conda env create -f numorphunet.yml

# Copy the package into the container
COPY . /app/

# Install the package in development mode
RUN /bin/bash -c "source activate 3dunet && pip install -e ."

# Create directory for models
RUN mkdir -p /models

# Add environment variables for better Nextflow compatibility
ENV PATH=/opt/conda/bin:/opt/conda/condabin:$PATH
ENV PYTHONUNBUFFERED=1
ENV CONDA_DEFAULT_ENV=3dunet
ENV PYTHONDONTWRITEBYTECODE=1

# Remove ENTRYPOINT to make it work with Nextflow
# Instead, set a default CMD that activates the conda environment
CMD ["/bin/bash", "-c", "source activate 3dunet && exec /bin/bash"]
