# Setup instructions

- Clone the Repo:

```bash
git clone https://openbeagle.org/aryan_nanda/gsoc_2024-enhanced_media_experience_with_ai-powered_commercial_detection_and_replacement.git
```

- Move to the cloned folder:

```bash
cd ~/gsoc_2024-enhanced_media_experience_with_ai-powered_commercial_detection_and_replacement
```

- In order to run the Yt8m dataset .ipynb files, first create a conda environment with necessary dependencies:

```bash
conda env create -f environment.yaml
```

- Then Activate the enviroment:

```bash
conda activate model
```

- To download 1/100-th of the training data from the US use:

```bash
curl data.yt8m.org/download.py | shard=1,100 partition=2/frame/train mirror=us python
```

> If you are from Asia use mirror=asia instead of us.

- Now, open the .ipynb and start running the cells.
