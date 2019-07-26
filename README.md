# OrcaML
1. pip install  **requirements.txt**
---

### Instructions for Testing:
1. Download **medium_orca.mdl** from [here](https://drive.google.com/drive/folders/1UOMcRVwGfyUZWrBFnGzRqcPcuS2vXbxi?usp=sharing).
- Put **medium_orca.mdl** in **models/**
- Run:  
**python3 process_wavs.py /path/to/data/**  
on 2 second long snippets of whale/background sounds. This will create a folder **/path/to/data/proccessed/** with resampled wavs and spectrograms in .npy format ready to be input into the model.
- Run **Train Model** and **Test SKRW** cells in **OrcaML.ipynb**, making sure to update paths to point to the **processed/** folder from step 3.
---

### Instructions for Training:
1. Process data as in step 3. of **Instructions for Testing**.
2. Place data into **data/** structured as shown in the first cell of **OrcaML.ipynb**
3. Uncomment training in **Train Model** cells of **OrcaML.ipynb** and run.
