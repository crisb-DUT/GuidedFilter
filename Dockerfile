FROM mcdenoising:latest

RUN apt-get install -y \
    libopenexr-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
    numpy \
    scipy 


ADD external /external
WORKDIR /external/openexrpython
RUN python setup.py install

VOLUME /workspace
VOLUME /workspace/data
WORKDIR /workspace

