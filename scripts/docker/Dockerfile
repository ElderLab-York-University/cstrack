# Build script for Docker image.
#
# Author: Helio Perroni Filho

FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

USER root
WORKDIR /
SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

# Build settings
ARG uid
ARG gid
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto

# Create ordinary user.
RUN groupadd -g $gid user
RUN useradd -u $uid -g $gid -G dialout,plugdev,video -m -s /bin/bash user

# Copy scripts to image.
COPY --chown=user:user user.bashrc /home/user/.bashrc

# Setup the system-wide environment.
ADD build_root.sh .
RUN /bin/bash -e -x build_root.sh && rm build_root.sh

USER user
WORKDIR /home/user
CMD ["/bin/bash"]

# Setup the ordinary user environment.
ADD build_user.sh requirements_1.txt requirements_2.txt ./
RUN /bin/bash -e -x build_user.sh \
  && rm -f build_user.sh requirements_1.txt requirements_2.txt
