# Use Ubuntu 22.04 for better ARM64 support
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python and LaTeX
RUN apt-get update && apt-get install -y \
    curl \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    locales \
    texlive-xetex \
    texlive-luatex \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-fonts-extra \
    texlive-bibtex-extra \
    biber \
    latexmk \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /workspaces/non-avian-ml