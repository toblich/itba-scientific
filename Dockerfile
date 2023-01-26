# FROM continuumio/miniconda3:22.11.1

# WORKDIR /itba

# COPY ./config/environmentw.yml environmentw.yml

# RUN conda env update --name mne3 --file environmentw.yml
# RUN conda activate mne3

# ENTRYPOINT [ "python" ]


FROM python:3

RUN pip install --no-cache-dir \
  pandas \
  numpy \
  matplotlib \
  requests

ENTRYPOINT [ "python" ]
