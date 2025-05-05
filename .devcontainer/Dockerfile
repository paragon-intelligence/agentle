FROM python:3.13.2-bookworm

# Configure apt and install packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    git \
    zsh \
    ripgrep \
    sudo \
    locales \
    ca-certificates \
    unzip \
    wget \
    build-essential \
    make \
    libcairo2-dev \
    libgl1-mesa-dev \
    libmagic-dev \
    pkg-config \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup locale for better terminal rendering
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Set working directory
WORKDIR /workspace

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Create the vscode user to maintain compatibility with scripts that might expect it
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode \
    && echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Install Starship prompt for better appearance
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y

RUN pip install --root-user-action=ignore -U uv

# Install Oh My Zsh for root with better theme
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install useful plugins for zsh
RUN git clone https://github.com/zsh-users/zsh-autosuggestions /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting /root/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

# Configure zsh with a better-looking theme 
RUN echo 'export ZSH="/root/.oh-my-zsh"' > /root/.zshrc && \
    echo 'ZSH_THEME="robbyrussell"' >> /root/.zshrc && \
    echo 'plugins=(git python pip docker zsh-autosuggestions zsh-syntax-highlighting)' >> /root/.zshrc && \
    echo 'source $ZSH/oh-my-zsh.sh' >> /root/.zshrc && \
    echo 'alias ll="ls -la"' >> /root/.zshrc && \
    echo 'alias py="python"' >> /root/.zshrc

# Add custom prompt setting
RUN echo 'export PS1="🐍 Agentle:%~ $ "' >> /root/.zshrc

# Adicionar UV path para quando ele for instalado pelo post-create.sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Configure bash with nicer prompt (as fallback)
RUN echo 'export PS1="\\[\\033[01;32m\\]Agentle\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]$ "' >> /root/.bashrc

# Stay as root for operations
USER root