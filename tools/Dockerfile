#docker run -it -v (pwd):/workdir --rm 
FROM sameli/manylinux2014_x86_64_cuda_11.8
#RUN wget https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
COPY ./torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl .
RUN python3.11 -m pip install pybind11 ./torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl
RUN yum install -yy zip
# copy code, but fix setup.py paths 
COPY build.sh /build.sh
