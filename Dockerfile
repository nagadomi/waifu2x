FROM kaixhin/cuda-torch:7.5

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
  libsnappy-dev \
  graphicsmagick \
  libgraphicsmagick1-dev \
  libssl-dev \
  ca-certificates \
  git && \
  rm -rf /var/lib/apt/lists/*

RUN \
  luarocks install graphicsmagick && \
  luarocks install lua-csnappy && \
  luarocks install md5 && \
  luarocks install uuid && \
  luarocks install csvigo && \
  PREFIX=$HOME/torch/install luarocks install turbo

# suppress message `tput: No value for $TERM and no -T specified`
ENV TERM xterm

COPY . /root/waifu2x

WORKDIR /root/waifu2x
