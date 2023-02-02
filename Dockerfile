FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN pip3 install pandas
RUN pip3 install tqdm

WORKDIR /mnt/lime/homes/bjc53/dog_box

COPY . .

CMD ["python", "./learn.py"]