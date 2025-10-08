FROM hdgigante/python-opencv:4.11.0-ubuntu

ARG DEBIAN_FRONTEND=noninteractive

# Build deps for CGAL, PCL, Ceres, Boost, etc.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
    pkg-config \
    libeigen3-dev \
    libboost-all-dev \
    libcgal-dev \
    libpcl-dev \
    libceres-dev \
    libsuitesparse-dev \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
