# Build
from top level transcriptomic_clustering directory

`docker build -f docker/tc/Dockerfile -t tc .`


# Run with docker
To run, you'll need three things: your script, a folder containing input files, including selected genes and normalized anndata (or unnormalized one if you're including the normalization step), and a temporary directory where the program can save AnnData files (as a rule of thumb, make sure you have 2x the size of the AnnData file in available disk space). You'll need to mount the python script, anndata file, and temporary directory.

`docker run \
-v /local1/marmot/matt_dev/transcriptomic_clustering/scripts/run_iter_clust_docker.py:/mnt/scripts/run_iter_clust.py:ro \
-v /localssd/marmot/matt_dev/tc_data:/mnt/adata:ro \
-v /localssd/marmot/matt_dev/tc_data/tmp_data/MacoskoTmp:/mnt/tmp \
-v /localssd/marmot/matt_dev/tc_data/output/macosko:/mnt/output \
tc
`

Replace the first -v line with the script you want to run (should be based off of scripts/run_iter_clust_docker.py) - the script should be copied to /mnt/scripts/run_iter_clust.py otherwise docker run won't work.
Replace the second -v line with the folder containing the normalized h5ad file and the rm.eigen file
Replace the third -v line with a path to a temporary directory where temporary files can be written
Replace the fourth -v line with a path to where an output folder can be written