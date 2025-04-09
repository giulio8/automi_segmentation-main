FROM python:3.10.6

## DO NOT EDIT these 3 lines. (Sorry for editing)
RUN mkdir /automi_seg
# we mount the code directory instead of copying it
# COPY ./ /automi_seg
WORKDIR /automi_seg

EXPOSE 8000

## Install your dependencies here using apt install, etc.
## Include the following line if you have a requirements.txt file.

# Solo requirements!
COPY requirements.txt /automi_seg
RUN pip install -r requirements.txt



RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV PYTHONPATH "${PYTHONPATH}:/automi_seg/evaluation"
