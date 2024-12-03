# Spectrogrand
![Architecture Diagram](./docs/static/Spectrogrand_Architecture_Diagram.png)

---
**Spectrogrand: Computational Creativity Driven Audiovisuals' Generation From Text Prompts** has been accepted as a full research paper at the [Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP) 2024](https://icvgip.in/)! 🎉

---

## About the project
[Spectrograms](https://en.wikipedia.org/wiki/Spectrogram#:~:text=A%20spectrogram%20is%20a%20visual,sonographs%2C%20voiceprints%2C%20or%20voicegrams.) are **visual representations of audio samples** often used in Engineering applications as features for various downstream tasks. We unlock the **artistic value of spectrograms** and use them in both scientific and artistic domains through Spectrogrand: a pipeline to generate **interesting melspectrogram-driven audiovisuals** given text topic prompts. We also bake in lightweight **domain-driven computational creativity** assessment throughout steps of the generation process.  

In this regard, this pipeline has the following steps:
- We use [audioldm2-music](https://huggingface.co/cvssp/audioldm2-music) to generate multiple candidate house music songs for the topic text prompt. We then estimate each candidate's **novelty** from human-generated house music songs (collected from the [HouseX](https://github.com/Gariscat/HouseX) dataset) and **value** through its danceability score calculated using [Essebtia](https://essentia.upf.edu/models.html). We select the song with the highest equiweighted score for our pipeline.
- Then, we generate melspectrograms for the song as a whole, and for periodic chunks of the sample. These numerous images convey local intensity and temporal diversity scattered throughout different zones of the song.
- We use the parent spectrogram to deduce the genre of the song. Our [Resnet-101 based-model with augmented train-time transforms](./research/models/genre_classification.py) is the current SOTA on the `HouseX-full-image` task 🥳
- We then use [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) to generate candidate album covers for this song. The selected genre defines and augments the prompts through selecting base colours and [descriptor words](./public/housex-processing/corpus). We then estimate each candidate's **value** and **surprisingness** based on its aestheticness, and how likely it can fool a strong custom classifier (trained on human-generated and AI-generated album covers) into believing that the candidate is more human-generated. We select the image with the highest equiweighted score for our pipeline.
- We then use [magenta](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2) to perform arbitrary image style transfer on the selected album cover image and each of the song chunk's melspectrograms.
- At the end of the pipeline, one can hence generate a static video and two spectrogram-driven audiovisual videos. As an additional feature ✨, we also support [
stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) to automatically generate a music video of arbitrary length conditioned on the chosen album cover image.

---

## Key contributions
- 📄 New [corpus](./public/housex-processing/corpus/) with [EDMReviewer](https://edmreviewer.com/) reviews for [selected artists from the HouseX dataset](./public/housex-processing/selected_artists.txt) with [LLM-generated descriptor words](./public/housex-processing/llm-outputs/).
- 📄 New [dataset](./public/mumu-processing/album-source-classification/) of AI-generated and human-generated album covers for Dance music, as extracted from the [MuMu dataset](https://www.upf.edu/web/mtg/mumu), and an accompanying strong lightweight [Mobilenet-v3 based classifier model](./research/models/surprise_estimation.py).
- 📌 Novel computational creativity estimation pipeline for audiovisuals' generation involving `Novelty, Creativity, and Value` distributed across different modalities viz. `{audio, image}`.

---

## Getting Started
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 512" width=5% height=5%><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M304.2 501.5L158.4 320.3 298.2 185c2.6-2.7 1.7-10.5-5.3-10.5h-69.2c-3.5 0-7 1.8-10.5 5.3L80.9 313.5V7.5q0-7.5-7.5-7.5H21.5Q14 0 14 7.5v497q0 7.5 7.5 7.5h51.9q7.5 0 7.5-7.5v-109l30.8-29.3 110.5 140.6c3 3.5 6.5 5.3 10.5 5.3h66.9q5.3 0 6-3z"/></svg>

To run the pipeline on Kaggle, please review the instructions listed in [the Kaggle data release](https://www.kaggle.com/datasets/vijayjaisankar/spectrogrand-public-release/) and check out [the notebook](https://www.kaggle.com/code/vijayjaisankar/spectrogrand-pipeline-official/) (also linked in the dataset). 

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 512" width=5% height=5%><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M128 32C92.7 32 64 60.7 64 96V352h64V96H512V352h64V96c0-35.3-28.7-64-64-64H128zM19.2 384C8.6 384 0 392.6 0 403.2C0 445.6 34.4 480 76.8 480H563.2c42.4 0 76.8-34.4 76.8-76.8c0-10.6-8.6-19.2-19.2-19.2H19.2z"/></svg>

To run the pipeline locally, please follow the steps detailed in [this notebook](./spectrogrand_pipeline.ipynb). 

---

## Outputs of Spectrogrand

---

####  Prompt topic: `Futuristic Spaceship`

**Static video**

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/e2782e91-9a3d-4e00-b610-b119395cd872

**Dynamic videos**

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/112ee67a-dc27-4d87-ad06-78bb3104f4c8

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/1c69895e-1764-4a45-92be-553949d5af47

---

####  Prompt topic: `Dystopian Robotic World`
(💡 Inspiration from [Twitter](https://twitter.com/punpeddler_/status/1766461639476588729))

**Static video**

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/be0f32e5-c4f5-4b9d-88a3-2a5cc65dd30e


**Dynamic videos**

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/3906e5a2-549c-4851-a81c-d7f5dd2b3f29

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/3fb2ed9b-d694-49e7-904c-7422038d7f20

---

####  Prompt topic: `Computer Vision`

**Static video**

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/bf373d21-9a05-4264-9637-e75f729008f8


**Dynamic videos**

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/4fd9b01b-4299-4f26-8e0e-6b1d1bc4c844

https://github.com/vijay-jaisankar/spectrogrand/assets/56185979/355318aa-514a-4f06-88d6-92dcad099760

---

## Acknowledgements and Contact Details
This project was done under the guidance of [Prof. Dinesh Babu Jayagopi](https://www.iiitb.ac.in/faculty/dinesh-babu-jayagopi).

Corresponding email: vijay.jaisankar@iiitb.ac.in
