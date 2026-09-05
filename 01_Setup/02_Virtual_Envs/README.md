# Setting Up Python and Virtual Environments


## Modules in Euler

module load 1) stack/2024-05   2) gcc/13.2.0   3) cuda/13.0.2   4) python/3.11.6_cuda   5) code-server/4.12.0   6) eth_proxy

mkdir venvs
mkdir venvs/ngioteaching_2026

cd venvs
python -m venv ngioteaching_2026
source ngioteaching_2026/bin/activate 

pip install ngio>=1.0 matplotlib ipykernel ez-zarr

## Python


## Virtual Environments



$HOME/.config/euler/jupyterhub/jupyterlabrc

with the content:

module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy hdf5/1.14.3



python -m ipykernel install --user --name "ngio_teaching_2026" --display-name "Python (ngio_teaching)2026)"
