## Repository for publication "Distinct neural representations encode psychiatric states across multiple timescales"

Preprint: https://www.medrxiv.org/content/10.64898/2026.07.30.26359255v1 

This repository includes code for: 
 - pre-processing Medtronic RC+S data: dependency [analysis-rc-s-data](https://github.com/openmind-consortium/Analysis-rcs-data)
 - data artifact annotation and plotting  - neural feature computation: dependency [py-neuromodulation](https://github.com/neuromodulation/py_neuromodulation)
 - audio feature computation using [OpenSmile](https://github.com/audeering/opensmile) and [Wave2Vec2](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
 - video Facial Action Unit computation using [OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU)

Preprocessing code is separated into:
 - ERP Momentary distress analysis: [preprocessing/erp](preprocessing/erp)
 - OCD symptom severity YBOCS analysis: [preprocessing/resting-state](preprocessing/resting-state)

Analaysis scripts are saved per figure in [plotting](plotting).

The required dependencies are listed in `pyproject.toml` and can be installed in a python virtual environment using [uv](https://docs.astral.sh/uv/).

Contact: Timon Merk (timon.merk@bcm.edu) and PI Nicole Provenza (Nicole.Provenza@bcm.edu).
