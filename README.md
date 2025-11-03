# Instruction to run

The project was developed using the environment created by the Dockerfile at the beginning of the semester (at USC).

### Another Options provided

- Using `requirements.txt` and a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Using `environment.yml` to reproduce a conda environment

```bash
conda env create -f environment.yml
conda activate pos-tagger
```

Both of these options work as well, although they might not have the same packages versions as the Docker environment.

### Choices

- Run `metric_testing.ipnyb` for the full experiment on models. Takes a long time to run (Trains +40 models)

- Run `BiLSTM_inference.ipynb` for a short test in 3 languages whith a configurable model.
