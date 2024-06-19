# Setup instructions

- Clone the Repo:

```bash
git clone https://openbeagle.org/aryan_nanda/gsoc_2024-enhanced_media_experience_with_ai-powered_commercial_detection_and_replacement.git
```

- Move to the cloned repo:

```bash
cd ~/gsoc_2024-enhanced_media_experience_with_ai-powered_commercial_detection_and_replacement/
```

- Move to the dataset folder:

```bash
cd datasetCollection_And_featureExtraction/yt8m
```

- In order to run the Yt8m dataset .ipynb files, first create a conda environment with necessary dependencies:

```bash
conda env create -f environment.yaml
```

- Then Activate the enviroment:

```bash
conda activate model
```

- Create a new folder for downloading dataset, the .ipynb files expect it to be at `2/frame/train`. (You can change it if you want.)

```bash
mkdir -p 2/frame/train
```

- Now, Download the vocabulary.csv from [here](https://research.google.com/youtube8m/csv/2/vocabulary.csv). (Vocabulary.csv is used to map video_id to its label name)

- After downloading it, move it to the folder `2/frame/`.

```
mv ~/Downloads/vocabulary.csv ./2/frame/vocabulary.csv
```

- Now, Download 1/100-th of the training data from the US server in the `2/frame/train` folder:

```bash
cd 2/frame/train
curl data.yt8m.org/download.py | shard=1,100 partition=2/frame/train mirror=us python
```

> If you are from Asia use mirror=asia instead of us.

- Now, open the .ipynb and start running the cells.
