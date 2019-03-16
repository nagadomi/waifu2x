FROM nagadomi/torch7:cuda10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
  libsnappy-dev \
  graphicsmagick \
  libgraphicsmagick1-dev \
  libssl1.0-dev \
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
