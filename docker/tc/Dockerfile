FROM python:3.9.16-buster

RUN python -m pip install --upgrade pip

WORKDIR /tc
ADD transcriptomic_clustering ./transcriptomic_clustering
ADD scripts ./scripts
ADD requirements.txt ./requirements.txt
ADD setup.py ./setup.py
ADD README.md ./README.md

RUN ls

RUN python3 -m venv tc_env
RUN . tc_env/bin/activate

RUN mkdir -p /mnt/tmp/
RUN mkdir -p /mnt/scripts/
RUN mkdir -p /mnt/adata/

RUN pip install numpy
RUN pip install -r requirements.txt && pip install -e .

CMD ["python", "/mnt/scripts/run_iter_clust.py"]