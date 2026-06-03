---
title: Non-Avian Acoustic Classifier Development
subtitle: A brief report for CDFW
authors:
  - name: Amy Van Scoyoc, PhD
    email: avanscoyoc@berkeley.edu
    orcid: 0000-0001-8638-935X
abstract: |
  The rapid proliferation of deep learning models has enabled researchers to analyze large volumes of passively collected acoustic data for wildlife monitoring. While robust models exist for commonly recorded species with abundant labeled training data (e.g., bird species), rare or data-deficient species often lack sufficient high-quality reference audio recordings for effective classification of field data. Here, I labeled >6500 audio clips from raw data and used few-shot transfer learning to tune and develop binary classifiers for 23 non-avian sound classes. To ensure robust model classification, I benchmarked performance across five model architectures and training sizes and highlight best practices on the amount of audio to use to achieve optimal model performance. Then, I trained a multi-class model with several anthropogenic sound classes to enhance the ability to detect anthropogenic sounds with greater accuracy. Finally, after reviewing more than 10,000 hours of audio with a team of fourteen annotators I surfaced the need for a rapid, inline annotation tooling. As an additional deliverable, our team developed Jupyter Bioacoustics to support data labeling and model training going forward.
---

## Section 1: Introduction and Project Scope

### 1.1 Background
Passive acoustic monitoring (PAM) is a non-invasive, scalable tool for wildlife monitoring. While deep learning models like BirdNET excel at avian classification, they often struggle with anthropogenic, abiotic, and non-avian biological sounds. This project addresses these gaps by developing custom classifiers for priority sound classes identified by CDFW. 
I produced 23 sound classifiers, including 7 novel models for priority categories, and conducted a systematic benchmarking experiment across five architectures and various training sizes to identify optimal training size and architecture for each class. Additionally, I developed best-practice guidance for ongoing monitoring and co-designed Jupyter Bioacoustics, an open-source annotation tool to streamline the labeling of audio clips for training new classifiers or for assignment validation.

### 1.2 Agency request
Target categories included a Human Noise Index (aggregating engine, gunshots, sirens, etc.), specific sources like generators and traffic, and environmental factors like device static, wind, and flowing water. Biotic priorities included the Nutria, coyote, Foothill yellow-legged frog.
These requests collectively span the full acoustic range of the monitoring program: short anthropogenic events, continuous abiotic noise, chorusing biological signals, and discrete vocalizations from data-deficient taxa.

### 1.3 Project objectives
Objectives were to assemble training libraries, identify the best architecture for each class, and provide the agency with validated models and minimum training-data requirements.

## Section 2: Data Collection and Annotation

### 2.1 Source recordings
Audio was sourced from CDFW ARU deployments (SoundHub API), Mojave Desert field sites, and external contributors (Ryan Peek, Judas Nutria team, Brashares Lab). All audio clips were field recorded on AudioMoth or SongMeter ARUs and mean transformed to mono input where applicable.

### 2.2 Clip selection and labeling workflow
I and fourteen additional reviewers annotated over 10,000 hours of audio for this project. Clips were selected in one of three ways: margin sampling, de novo labeling, or clip sharing. 
First, for existing classifiers that needed to be tuned to improve performance (e.g., coyote, bullfrog, engine) I used margin sampling on BirdNET predictions at all CDFW sampling sites. I sampled 200–300 audio segments for each sound class across the full range of BirdNET prediction scores (0.10–1.0), along with an additional 200–300 segments from higher-confidence predictions (score > 0.5) to facilitate upsampling. 
Second, for novel classifier categories, I selected 600-800 1-minute audio clips based on site-level covariates at CDFW sampling sites (distance to built environment, stream proximity, recordings had device static, etc.) as stratified by time and location.
Third, I used verified clips. These 1-10 minute audio clips were contributed by the CDFW team (Ryan Peek's foothill yellow-legged frog dataset; the Judas Nutria team's nutria recordings) and the Brashares Lab Cannabis project at UC Berkeley and were annotated to 3 second clips. 
Each audio segment was annotated as either positive (target sound present) or negative (absent) by two independent reviewers. Disagreements were resolved through adjudication by a third independent reviewer. Annotations were conducted using Whombat, Raven Pro, and a custom python Tkinter application I built to compare approaches. 
Eight classes reached >400 positive clips (e.g., wind, generator), enabling full benchmarking. Six classes had 100-300 clips (e.g., traffic, nutria), while nine remained data-deficient for the benchmarking experiment (<60 clips) but were still used to create single and multi-class models.

### 2.3 Jupyter Bioacoustics: annotation software development
The existing annotation tools (Raven, Whombat, and our python Tkinter executable) proved inefficient for cloud-resident data and distributed teams. As a result, we built Jupyter Bioacoustics to embed annotation directly into the JupyterLab environment, ensuring standardized review and a traceable audit trail for collaboration. The tool features inline spectrograms, YAML-configured forms, and flexible loading from S3/GCS. While it was not used in this project’s annotations, our design, inspired by this project is now publically available as an open-source resource.

## Section 3: Model Benchmarking Experiment
The 23 target sound classes in this project span a wide range of acoustic structures: chorusing amphibians and insects with overlapping vocalizations (Pacific chorus frog, field cricket), long discrete calls (coyote), short impulsive events (gun, fireworks), broadband transient anthropogenic sources (engine, power tools), and continuous abiotic noise (wind, device static, flowing water). No single pre-trained classifier architecture is known to perform uniformly well across this diversity, and the appropriate choice of model may depend both on the structure of the target call and on the amount of training data available.

Most bioacoustics classifiers in current use, such as BirdNET and Google’s Perch, were primarily pre-trained with bird vocalizations. While there is some evidence to suggest that these models transfer to amphibians and mammals (Ghani et al. 2023), they may not transfer equally well to anthropogenic disturbance, or abiotic sounds. General-purpose audio architectures (ResNet, MobileNet, VGG) offer an alternative but lack domain-specific pre-training. I therefore conducted a systematic benchmarking experiment to (1) identify the best-performing architecture for each target class, and (2) characterize how performance scaled with the size of the labeled training set to select the best model. The performance curves from this benchmarking experiment were used to select the best model architecture to train the final classifiers.

### 3.2.1 Model architectures compared

To test whether model architecture influences performance across these sound types, we benchmarked five architectures (BirdNET, Perch, ResNet-18, MobileNet v2, VGG-11) to identify the best performers and determine how performance scales with training size for each call structure. I evaluated five convolutional neural network architectures, selected to span bioacoustics-specific pre-training and general-purpose audio/image pre-training:

- **BirdNET 2.4** — a pre-trained bioacoustics classifier; trained mainly on bird vocalizations.
- **Perch v8** — a more recent bioacoustics foundation model from Google Research, also pre-trained on bird vocalizations but with a different embedding architecture.
- **ResNet-18** — a general-purpose residual network architecture applied to spectrograms.
- **MobileNet v2** — a lightweight architecture similar to BirdNET, without pretraining.
- **VGG-11** — a classical deep CNN architecture, included as a baseline.

For all five architectures, I used the pre-trained backbone as a frozen feature extractor and trained a linear classifier (multilayer perceptron head) on top of the embeddings for 20 epochs with a batch size of 32.

### 3.2.2 Sound classes and training libraries
We used stratified 5-fold cross-validation on 13 classes, with balanced positive and negative training sizes from 10 to 160 samples (Table 1). Audio clips were standardized to 3 seconds for all models, with an additional silent padding up to 5 seconds specifically for Perch. The negative training set for a given class consisted of positive samples evenly drawn from all other sound classes. All trained classifiers at each combination of training size and model were evaluated on a fixed, held-out test set of 50 positive and 50 negative clips for that specific sound class. The primary performance metric was the area under the receiver operating characteristic curve (AUC-ROC), which summarizes the trade-off between true and false positive rates across all possible decision thresholds. AUC-ROC of 1.0 indicates perfect ranking of positive over negative examples; 0.5 indicates random performance.
**Table 1.** Sound classes and library composition for the benchmarking experiment.

| Sound class | Positive clips | Negative clips | Max training size | Audio clip origin |
|---|---|---|---|---|
| Device static | 1,416 | 1,416 | 160 | Opportunistic, CDFW notes |
| Pacific chorus frog | 795 | 795 | 160 | Margin sampling, Mojave |
| Woodhouse's toad | 638 | 638 | 160 | Margin sampling, Mojave |
| Field cricket | 605 | 605 | 160 | Margin sampling, Mojave |
| Wind | 508 | 508 | 160 | Opportunistic |
| Generator | 437 | 437 | 160 | Brashares cannabis project |
| Foothill yellow-legged frog | 436 | 436 | 160 | Ryan Peek, CDFW |
| American bullfrog | 421 | 421 | 160 | Margin sampling, Mojave |
| Traffic | 297 | 297 | 160 | Site covariates, opportunistic |
| Nutria | 245 | 245 | 160 | Judas Nutria team, CDFW |
| Engine | 170 | 170 | 80 | Site covariates, opportunistic |
| Power tools | 162 | 162 | 80 | Margin sampling, opportunistic |
| Human vocal | 117 | 117 | 40 | Margin sampling, opportunistic |
| Coyote | 101 | 101 | 40 | Margin sampling, Mojave |


Nine additional classes (metal, airplane, thunder, gun, fireworks, dog, water, human non-vocal, branches) had insufficient labeled data to be included in the benchmarking experiment (≤60 clips per polarity) and were excluded from the architectural sweep. These classes were either trained on later using the architecture indicated for acoustically similar classes, or flagged as data-deficient.

**Table 2.** Additional nine sound classes not included in the experiment.

| Sound class | Positive clips | Negative clips | Audio clip origin |
|---|---|---|---|
| Metal clanging | 60 | 60 | Opportunistic |
| Airplane | 52 | 52 | Opportunistic |
| Thunder | 21 | 21 | Opportunistic |
| Gun | 19 | 19 | Margin sampling, all CDFW sites |
| Fireworks | 13 | 13 | Margin sampling, all CDFW sites |
| Dog | 12 | 12 | Opportunistic |
| Water | 11 | 11 | ARU location info, stream likely; opportunistic |
| Human non-vocal | 8 | 8 | Margin sampling, all CDFW sites |
| Branches | 3 | 3 | Opportunistic |



## Section 4: Classifier Development and Results

### 4.1 Results

:::{figure} ../figs/species_comparison.png
:label: fig-species-comparison
:align: center
AUC-ROC performance across all 23 sound classes and five model architectures. Each panel shows performance curves as a function of training size. Bioacoustics-pretrained models (BirdNET, Perch) generally outperform general-purpose architectures for biological sounds.
:::

Results indicated that bioacoustics-pretrained models (BirdNET, Perch) excel for biological sounds, while general architectures are comparable for anthropogenic noise. Performance generally saturates near 80 samples for BirdNET but is outperformed by Perch beyond 160. Cross-class analysis shows how rapidly each architecture's performance saturates with training size. The best architectures were retrained on full libraries. High performance was achieved for abiotic (wind, static) and biological (frogs, crickets) classes. Data-deficient classes (thunder, dog) remain preliminary.

:::{figure} ../figs/_frog_comparison.png
:label: fig-frog-comparison
:align: center
AUC-ROC by training size for frog and toad classes (American bullfrog, Pacific chorus frog, Woodhouse's toad, foothill yellow-legged frog). Bioacoustics-pretrained models show strong performance even at low training sizes for these chorusing taxa.
:::

:::{figure} ../figs/_nonfrog_comparison.png
:label: fig-nonfrog-comparison
:align: center
AUC-ROC by training size for non-frog biological and abiotic classes (coyote, field cricket, wind, device static, nutria). Architecture performance patterns differ notably from frog classes, with general-purpose models more competitive for abiotic sounds.
:::

:::{figure} ../figs/_anthro_comparison.png
:label: fig-anthro-comparison
:align: center
AUC-ROC by training size for anthropogenic sound classes (engine, generator, traffic, power tools, human vocal). General-purpose architectures (ResNet-18) are competitive with bioacoustics-pretrained models for these non-biological sounds.
:::

#### Representative individual-species results

:::{figure} ../figs/results_nutria_plot.png
:label: fig-nutria
:align: center
Benchmarking curves for nutria. With 245 positive clips and a distinctive low-frequency call, all architectures reached high AUC-ROC, with Perch leading at larger training sizes.
:::

:::{figure} ../figs/results_yellow_legged_frog_plot.png
:label: fig-ylf
:align: center
Benchmarking curves for foothill yellow-legged frog. Performance scales steeply with training size; Perch and BirdNET outperform general architectures reflecting the biological pre-training advantage for this chorusing species.
:::

:::{figure} ../figs/results_coyote_plot.png
:label: fig-coyote
:align: center
Benchmarking curves for coyote. With only 101 positive clips (max training size 40), BirdNET achieves the highest performance, likely benefiting from pre-training on mammal vocalizations.
:::

:::{figure} ../figs/results_device_static_plot.png
:label: fig-device-static
:align: center
Benchmarking curves for device static. As an abiotic signal, device static saturates quickly across all architectures; ResNet-18 and BirdNET reach near-perfect AUC-ROC by 40 samples.
:::

:::{figure} ../figs/results_wind_plot.png
:label: fig-wind
:align: center
Benchmarking curves for wind. Wind is a broadband continuous noise; general-purpose architectures perform comparably to bioacoustics-pretrained models, with all architectures approaching ceiling at 80 samples.
:::

### 4.2 Human Noise Index

A multiclass Human Noise Index covers ten classes, aggregating detections into site-level summaries of disturbance. While shared representations help, some confusion exists between similar engine sounds.

:::{figure} ../figs/confusion_matrix_binary_anthro_birdnet.png
:label: fig-confusion-anthro
:align: center
Confusion matrix for the binary anthropogenic noise classifier (BirdNET backbone). Rows indicate true class; columns indicate predicted class. Off-diagonal entries highlight the main sources of confusion between acoustically similar anthropogenic categories.
:::

## Section 5: Best-Practice Guidance for Acoustic Monitoring

Guidance includes: (1) use Perch/BirdNET for biotic sounds and ResNet for abiotic; (2) target 160 positive samples for training; (3) calibrate thresholds based on site-level noise covariates.

## Section 6: Future Work

Future work should prioritize labeling for data-deficient classes and investigate full backbone fine-tuning. Field validation is necessary to assess precision under real-world imbalanced prevalence.

## Section 7: Deliverables

Provided to CDFW: 23 classifiers, the Human Noise Index, labeled libraries, and benchmarking data. Jupyter Bioacoustics is released as an open-source extension on GitHub.

