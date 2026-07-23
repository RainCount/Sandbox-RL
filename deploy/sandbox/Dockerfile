FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl docker.io fuse-overlayfs git sudo \
    && rm -rf /var/lib/apt/lists/*

# Pin envd in production by replacing this installer with the release artifact
# approved for your AGS account. Kept compatible with the existing working template.
RUN curl -fsSL https://envd.dev | bash

RUN mkdir -p /home/user /tmp/swe /var/lib/docker \
    && useradd -m -s /bin/bash user 2>/dev/null || true \
    && echo 'user ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/user

WORKDIR /home/user
EXPOSE 49983 49999
HEALTHCHECK --interval=10s --timeout=3s --retries=10 CMD command -v fuse-overlayfs >/dev/null || exit 1
CMD ["envd", "up"]

