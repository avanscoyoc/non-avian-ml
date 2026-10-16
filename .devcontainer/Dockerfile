# Use Ubuntu 22.04 for better ARM64 support
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies: audio I/O, ffmpeg (scripts/make_window_clips.py) and LaTeX (pixi run brief / slides).
# Python and everything else come from pixi.
RUN apt-get update && apt-get install -y \
    curl \
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
    texlive-science \
    biber \
    latexmk \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /workspaces/non-avian-ml