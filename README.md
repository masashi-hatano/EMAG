
<div align="center">

# EMAG: Ego-motion Aware and Generalizable 2D Hand Forecasting from Egocentric Videos

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

**[[Paper](https://masashi-hatano.github.io/assets/pdf/emag.pdf)][[Supplementary](https://masashi-hatano.github.io/assets/pdf/emag_supp.pdf)][[Project Page](https://masashi-hatano.github.io/EMAG/)][[Poster](https://masashi-hatano.github.io/assets/pdf/emag_poster.pdf)]**

</div>

This is the official code release for our ECCVW 2024 paper \
"EMAG: Ego-motion Aware and Generalizable 2D Hand Forecasting from Egocentric Videos".

## System
- **OS**: `Linux / Ubuntu 20.04`
- **Python Version**: `Python 3.8.10`
- **CUDA Version**: `CUDA 11.7`

## 🔨 Installation
```bash
# Create a virtual environment
python3 -m venv emag
source emag/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

## 🔥 Training
```bash
python3 lit_main.py train=True test=False
```

## 🔍 Evaluation
To evaluate the model, please run the following command.
```bash
python3 lit_main.py train=False test=True
```

## ✍️ Citation
If you use this code for your research, please cite our paper.
```bib
@inproceedings{Hatano2024EMAG,
    author = {Hatano, Masashi and Hachiuma, Ryo and Saito, Hideo},
    title = {EMAG: Ego-motion Aware and Generalizable 2D Hand Forecasting from Egocentric Videos},
    booktitle = {European Conference on Computer Vision Workshops (ECCVW)},
    year = {2024},
}
```