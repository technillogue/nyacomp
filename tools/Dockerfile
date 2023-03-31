#docker run -it -v (pwd):/workdir --rm 
FROM sameli/manylinux2014_x86_64_cuda_11
#RUN wget https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
RUN python3.10 -m pip install pybind11 https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp310-cp310-linux_x86_64.whl
RUN yum install zip
# copy code, but fix setup.py paths 
COPY build.sh /build.sh
