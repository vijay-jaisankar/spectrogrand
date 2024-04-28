## Spectrogrand: Audio survey analysis

This document details the steps to be taken to create the data splits for analysing human preference of audio samples.

---

- First, download the PMQD dataset from the [official release](https://github.com/carlthome/pmqd/releases/download/v1.0.0/audio32.tgz). This file can be unzipped by running `tar -xvzf audio32.tgz`. After this step, please note the path of the directory containing the `.wav` files through `cd ./audio32 && ls *.wav | wc -l && pwd && cd ../`.
- Then, navigate to [pool-1](./survey-audios/pool-1/), and run `chmod +x copy_files.sh && ./copy_files.sh PATH_TO_AUDIO_FILES`, where *PATH_TO_AUDIO_FILES* is the path containing the downloaded `.wav` files.
- Then, navigate to [pool-2](./survey-audios/pool-2/), and run `chmod +x copy_files.sh && ./copy_files.sh PATH_TO_AUDIO_FILES`, where *PATH_TO_AUDIO_FILES* is the path containing the downloaded `.wav` files.