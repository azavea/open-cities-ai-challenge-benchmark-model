ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY ./benchmark /opt/src/benchmark

RUN pip install pystac==0.3.1
RUN pip install geopandas==0.6.1

CMD ["bash"]
