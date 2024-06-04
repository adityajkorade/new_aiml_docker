# Use an NVIDIA CUDA base image with CUDA 12.2 and cuDNN installed
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV NB_USER="gpuuser"
ENV UID=999
ENV DEBIAN_FRONTEND noninteractive

# Update system and install dependencies
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    ca-certificates \
    software-properties-common \
    locales \
    gcc pkg-config libfreetype6-dev libpng-dev g++ \
    pandoc \
    sudo \
    curl \
    libffi-dev \
    net-tools \
    rsync \
    vim \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Add deadsnakes PPA and install Python 3.9
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.9 python3.9-dev python3.9-distutils python3-pip gfortran libopenblas-dev liblapack-dev

# Update alternatives to use Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install Python packages
RUN python3.9 -m pip install --upgrade pip requests setuptools pipenv

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENV PATH=/usr/bin/python3.9:$PATH

ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER="${NB_USER}" \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    PATH="${CONDA_DIR}/bin:${PATH}" \
    HOME="/home/${NB_USER}"

USER root

RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -l -m -s /bin/bash -u $UID $NB_USER && \
    echo "$NB_USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    mkdir -p "${CONDA_DIR}" && \
    chown -R "${NB_USER}" "${CONDA_DIR}" && \
    chmod g+w /etc/passwd

USER ${NB_USER}

ENV PATH=$CONDA_DIR/bin:$PATH
ENV PATH=/home/$NB_USER/.local/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -f -b -p /opt/conda && rm -rf ~/miniconda.sh

RUN conda install -c conda-forge mamba python==3.9
RUN mamba install -y -q -c "nvidia/label/cuda-12.2.2" cuda-nvcc

USER ${NB_USER}

ENV PATH=$CONDA_DIR/bin:$PATH
ENV PATH=/home/$NB_USER/.local/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -f -b -p /opt/conda && rm -rf ~/miniconda.sh

RUN python3.9 -m pip install \
    bioblend==1.0.0 \
    galaxy-ie-helpers==0.2.7 \
    numba==0.56.4 \
    aquirdturtle_collapsible_headings==3.1.0 \
    jupyterlab-nvdashboard==0.7.0 \
    bokeh==2.4.0 \
    jupyter_server==1.16.0 \
    jupyterlab==3.4.6 \
    nbclassic==0.4.8 \
    jupyterlab-git==0.39.3 \
    jupytext==1.14.1 \
    jupyterlab-execute-time==2.3.0 \
    jupyterlab-kernelspy==3.1.0 \
    jupyterlab-system-monitor==0.8.0 \
    jupyterlab-topbar==0.6.1 \
    onnx==1.12.0 \
    onnx-tf==1.10.0 \
    tf2onnx==1.13.0 \
    skl2onnx==1.13 \
    scikit-image==0.19.3 \
    opencv-python==4.6.0.66 \
    nibabel==4.0.2 \
    onnxruntime==1.13.1 \
    seaborn==0.12.1 \
    voila==0.3.5 \
    elyra==3.14.1 \
    bqplot==0.12.36 \
    biopython==1.79 \
    torch==2.2.1 \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

RUN python3.9 -m pip install --no-deps \
    xformers==0.0.25.post1 \
    trl==0.8.6 \
    peft==0.10.0 \
    accelerate==0.30.0 \
    bitsandbytes[cuda]==0.43.1

USER root 

RUN ldconfig
RUN ldconfig /usr/lib64-nvidia
RUN ldconfig /usr/local/cuda-12.2.2

RUN mkdir -p /home/$NB_USER/.ipython/profile_default/startup/
RUN mkdir -p /import
RUN mkdir -p /home/$NB_USER/notebooks/
RUN mkdir -p /home/$NB_USER/usecases/
RUN mkdir -p /home/$NB_USER/elyra/
RUN mkdir -p /home/$NB_USER/data

COPY ./startup.sh /startup.sh
COPY ./get_notebook.py /get_notebook.py

COPY ./galaxy_script_job.py /home/$NB_USER/.ipython/profile_default/startup/00-load.py
COPY ./ipython-profile.py /home/$NB_USER/.ipython/profile_default/startup/01-load.py
COPY ./jupyter_notebook_config.py /home/$NB_USER/.jupyter/

COPY ./*.ipynb /home/$NB_USER/

COPY ./notebooks/*.ipynb /home/$NB_USER/notebooks/
COPY ./usecases/*.ipynb /home/$NB_USER/usecases/
COPY ./elyra/*.* /home/$NB_USER/elyra/

COPY ./data/*.tsv /home/$NB_USER/data/

ENV DEBUG=false \
    GALAXY_WEB_PORT=10000 \
    NOTEBOOK_PASSWORD=none \
    CORS_ORIGIN=none \
    DOCKER_PORT=none \
    API_KEY=none \
    HISTORY_ID=none \
    REMOTE_HOST=none \
    GALAXY_URL=none

RUN mkdir -p /import
RUN mkdir -p ${NB_USER}
RUN mkdir -p /home/${NB_USER}

RUN ls -lah /home/${NB_USER}/ && ls -lah /import/

# Change ownership of the directories to NB_USER
RUN chown -R ${NB_USER}:${NB_USER} /home/${NB_USER} /import

USER ${NB_USER}

WORKDIR /import

CMD /startup.sh
