---
title: Perch outperforms BirdNET under multiclass evaluation of non-biotic sounds
subtitle: Draft
authors:
  - name: Amy Van Scoyoc
    email: avanscoyoc@berkeley.edu
    orcid: 0000-0001-8638-935X
    website: https://github.com/avanscoyoc/non-avian-ml
abstract: |
  The rapid proliferation of deep learning models has enabled researchers to analyze large volumes of passively collected acoustic data for wildlife monitoring. While robust models exist for commonly recorded species with abundant labeled training data (e.g., common bird species), classifier performance for non-avian, rare, or data-deficient targets remains less well characterized. Most published benchmarks evaluate classifiers using binary AUC against generic background negatives, a setting that does not reflect real-world deployment in which models must discriminate among multiple acoustically related classes. Here, we compared the performance of five convolutional neural network (CNN) classifiers on field-collected audio from California, spanning four anuran species, two non-avian mammals, and seven anthropogenic and abiotic sound classes. We evaluated classifier performance under both binary and ecologically realistic multiclass settings within taxonomically coherent groups. We found that binary AUC overstates non-avian classifier performance, that bird-pretrained models (BirdNET, Perch) outperform image-pretrained baselines across all class types, and that Perch was substantially more robust than BirdNET under multiclass evaluation, particularly for anthropogenic and abiotic sounds. Based on these findings, we recommend that practitioners report multiclass evaluation within ecologically coherent groups, and consider Perch over BirdNET for deployments involving multi-class discrimination of non-avian targets.  
data_availability: |
  Labeled audio clips are archived on Zenodo at [doi:10.5281/zenodo.20534256](https://doi.org/10.5281/zenodo.20534256).
---

```{raw} latex
\captionsetup{font=footnotesize, skip=8pt}
```

## Introduction

Passive acoustic monitoring (PAM) using autonomous recording units (ARUs) has become a scalable and increasingly standard approach for biodiversity monitoring (Sugai et al. 2019).
The volume of recorded audio has outpaced the capacity for manual annotation, motivating the use of deep learning classifiers to automate detection. Convolutional neural network (CNN) classifiers, in particular pre-trained bioacoustic models such as BirdNET (Kahl et al. 2021) and Perch (van Merriënboer et al. 2025), are now widely deployed for species detection across taxonomic groups. These tools have expanded the practical reach of PAM into applications including biodiversity surveys, conservation monitoring of rare species, and characterization of anthropogenic disturbance (Stowell 2022).

Most published evaluations of bioacoustic classifiers focus on avian targets with abundant labeled training data, where existing tools perform well. Non-avian, rare, or cryptic taxa, along with anthropogenic and abiotic sound classes, remain poorly characterized in benchmarks (Ghani et al. 2023). A second limitation concerns evaluation methodology. Classifier performance is typically reported as binary AUC against generic background negatives, which captures the discriminability of a target sound from absence but does not reflect the multi-class acoustic scene that classifiers encounter in deployment. In a real deployment context, sensors are exposed to a variety of similar sound characteristics and must distinguish among multiple acoustically related sound sources, including taxonomically similar species and acoustically similar disturbance sources. This evaluation gap has practical consequences. Models selected on binary AUC may not be the best-performing models in deployment, and the magnitude of this discrepancy has not been systematically characterized for non-avian targets.

Recent work has demonstrated that bird-pretrained embeddings (BirdNET, Perch) transfer effectively to non-bird biological taxa including bats, marine mammals, and anurans (Ghani et al. 2023). This is a surprising result given that these models were trained almost exclusively on bird vocalizations, and it has led to the widespread practice of using BirdNET and Perch as off-the-shelf feature extractors for non-avian PAM. Less is known about whether these models transfer to anthropogenic and abiotic sound classes that are acoustically more distant from bird vocalizations. It also remains unclear whether different bird-pretrained models, which appear comparable under binary evaluation, differ meaningfully under deployment-realistic multiclass evaluation.

Here, we evaluate five CNN architectures (BirdNET, Perch, ResNet-18, MobileNet v2, VGG-11) on field-collected audio from California spanning ten non-avian sound classes, including four anuran species and six anthropogenic or abiotic noise classes. We characterize binary classifier performance using a training-size sweep from 10 to 160 annotated clips per class. We then evaluate multiclass classifier performance within two ecologically coherent groups, an anuran species group and a non-biotic noise group, to assess realistic performance for field deployment. We compare binary and multiclass results directly to quantify the gap between standard evaluation and deployment-realistic evaluation, and discuss implications for PAM benchmarking practice and model selection.

## Methods

**Source recordings and annotation.** Audio was sourced from CDFW ARU deployments (SoundHub API), Mojave Desert field sites, and external contributors including Ryan Peek, the Judas Nutria team, and the Brashares Lab. All recordings were field-collected on AudioMoth or SongMeter ARUs and mean-transformed to mono.

Clips were selected through three workflows. For existing classifiers requiring tuning (e.g., coyote, bullfrog, engine), we used margin sampling on BirdNET 2.4 predictions. We sampled 200–300 audio segments for each class across the full range of BirdNET prediction scores (0.10 to 1.0), along with an additional 200–300 segments from higher-confidence predictions (score above 0.5) to facilitate upsampling. For novel classifier categories, we selected 600 to 800 one-minute clips at CDFW sampling sites stratified by time and location, using site-level covariates such as distance to built environment, stream proximity, and presence of device static. For specialist classes, we used verified clips contributed by external collaborators. These included the foothill yellow-legged frog dataset, nutria recordings, and the Brashares Lab Cannabis Project, all annotated to three-second clips.

Each audio segment was annotated as either positive (target sound present) or negative (absent) by two independent reviewers. Disagreements were resolved through adjudication by a third independent reviewer. Annotations were conducted using Whombat, Raven Pro, and a custom Python Tkinter application. A subset of positive clips contained additional vocalizing taxa or co-occurring sound sources in the background. This was most notable for California chorus frog and arroyo or red-spotted toad, which co-occur and frequently call together. We discuss the implications of this contamination in the Discussion.
```{raw} latex
\noindent
```
**Class composition.** We compiled libraries for ten non-avian target classes spanning a range of acoustic structures. These included four chorusing anuran species (California chorus frog Pseudacris cadaverina, arroyo or red-spotted toad Anaxyrus californicus or A. punctatus, foothill yellow-legged frog Rana boylii, American bullfrog Lithobates catesbeianus) and six non-biotic classes representing anthropogenic and abiotic sound sources (engine, generator, traffic, power tools, device static, wind). Per-class positive sample sizes and audio sources are reported in Table 1.

```{raw} latex
\noindent
```
**Classifier training.** We compared five architectures spanning bioacoustic-specific and general-purpose pre-training. BirdNET 2.4 is a pre-trained bioacoustic classifier trained mainly on bird vocalizations. Perch v8 is a more recent bioacoustic foundation model from Google Research, also pre-trained on bird vocalizations but with a different embedding architecture. ResNet-18 is a general-purpose residual network applied to spectrograms. MobileNet v2 is a lightweight architecture similar to BirdNET, without bioacoustic pretraining. VGG-11 is a classical deep CNN architecture included as a baseline.

For all five architectures, we used the pre-trained backbone as a frozen feature extractor and trained a linear classifier (multilayer perceptron head) on top of the embeddings for 20 epochs with a batch size of 32. We evaluated the loss function using stratified 5-fold cross-validation. To generate confidence intervals each training condition was randomly sampled 10 times.

```{raw} latex
\noindent
```
**Binary classification.** For each of the ten classes, we trained a binary classifier on the corresponding model embeddings. Training set sizes ranged from 10 to 160 annotated positive clips per class, with variable upper bounds by class availability (**Table 1**). Training positives were paired 1:1 with negatives sampled evenly from the positive pools of the other thirteen classes. Audio clips were standardized to 3 seconds for all models, with an additional silent padding up to 5 seconds for Perch. All trained classifiers were tested on a held-out dataset of 50 positive and 50 negative samples per sound class, with the test set held constant across all training sizes and random seeds. Classifier performance was evaluated using Area Under the ROC curve (AUC-ROC), which summarizes the classifier's true positive rate against the false positive rate across all possible thresholds. Values closer to 1.0 indicate higher performance and values near 0.5 are no different from random.

:::{table} Sound classes and library composition for the benchmarking experiment.
:label: tbl-sound-classes

| Sound class | Positive clips | Negative clips | Max training size | Audio clip origin |
|---|---|---|---|---|
| Device static | 1,416 | 1,416 | 160 | Opportunistic, CDFW notes |
| Pacific chorus frog | 795 | 795 | 160 | Margin sampling, Mojave |
| Woodhouse's toad | 638 | 638 | 160 | Margin sampling, Mojave |
| Wind | 508 | 508 | 160 | Opportunistic |
| Generator | 437 | 437 | 160 | Brashares cannabis project |
| Foothill yellow-legged frog | 436 | 436 | 160 | CDFW, CDFW |
| American bullfrog | 421 | 421 | 160 | Margin sampling, Mojave |
| Traffic | 297 | 297 | 160 | Site covariates, opportunistic |
| Engine | 170 | 170 | 80 | Site covariates, opportunistic |
| Power tools | 162 | 162 | 80 | Margin sampling, opportunistic |
:::

```{raw} latex
\noindent
```
**Multiclass classification.** To assess deployment-realistic performance, we trained multiclass classifiers within two ecologically coherent class groups. The anuran group included four sympatric or partially sympatric anuran species (California chorus frog, arroyo or red-spotted toad, foothill yellow-legged frog, American bullfrog). The non-biotic group included six abiotic and anthropogenic noise classes (engine, generator, traffic, power tools, device static, wind).

For each group, we trained a single multiclass MLP head on top of frozen embeddings from BirdNET and Perch. Training used 80 positive examples per class drawn from the same training pools as the binary experiments, with 10 random seeds. The held-out test set comprised 50 positives per class. Multiclass performance was reported as per-class recall, confusion matrices, and per-class one-versus-rest AUC computed from the multiclass softmax outputs.

```{raw} latex
\noindent
```
**Binary versus multiclass comparison.** For each class in each group, we compared binary AUC at N equals 80 against multiclass one-versus-rest AUC at the same training size. Both metrics are AUC-based and use the same training pool and the same held-out test clips, which allows direct comparison of the discrimination provided by the same embeddings under the two evaluation regimes.

```{raw} latex
\noindent
```
**Embedding-space analyses.** To characterize the embedding-space geometry underlying multiclass confusions, we extracted Perch embeddings for approximately 100 randomly selected clips per class from each group's test pool and projected them to two dimensions using t-distributed stochastic neighbor embedding (t-SNE) with perplexity of 30. Class centroids were overlaid as reference points.

### Results

**Binary classifier evaluation.** Across all ten non-avian binary classifiers, BirdNET and Perch achieved binary AUC ≥ 0.85 at N=80, with Perch reaching saturation faster than BirdNET (N≈20 vs. N≈80). Image-pretrained CNN architectures (ResNet-18, MobileNet v2, VGG-11) did not reach competitive performance across any class type (**Figure 1**).

```{raw} latex
\newpage
```
:::{figure} ../figs/broadband.png
:label: fig-broadband
:align: center
Binary classifier AUC-ROC as a function of training size (N = 10–160 clips per class) for broadband anthropogenic and abiotic sound classes (device static, engine, generator, power tools, traffic, wind). Lines show mean AUC across 10 random seeds; shaded bands are 95% confidence intervals. BirdNET and Perch outperform the image-pretrained architectures (ResNet-18, MobileNet v2, VGG-11) at all training sizes.
:::

```{raw} latex
\newpage
```
:::{figure} ../figs/pulsed.png
:label: fig-pulsed
:align: center
Binary classifier AUC-ROC as a function of training size for the four anuran classes (American bullfrog, Pacific chorus frog, Woodhouse's toad, foothill yellow-legged frog). BirdNET and Perch reach near-ceiling performance (AUC ≥ 0.95) at N = 20–40 for most classes; image-pretrained architectures do not converge within the tested training range.
:::
  
```{raw} latex
\noindent
```
**Multiclass classifier evaluation.** For the multiclass anuran classifiers, Perch achieved 95 to 100% recall across all four anuran classes with ≤5% off-diagonal confusion, while BirdNET showed more variation with 70 to 100% recall (**Figure 2a**). Yellow-legged frog performed the best, achieving 100% recall in both BirdNET and Perch. The principal confusion was between Pacific chorus frog and Woodhouse’s toad (21% of PCF examples misclassified). Binary one-vs-rest AUC remained ≥0.95 in both models, showing a strong ranking despite operating-point errors.

For the multiclass abiotic classifiers, Perch’s multiclass recall spanned 68% (traffic) to 94% (power tools). Similarly, there was a greater spread in the BirdNET multiclass recall, with 20% for traffic to 79% for wind (**Figure 2b**). Between the two architectures, recall improvements were largest from BirdNet to Perch for engine (32% to 73%), device static (32% to 86%), and traffic (20% to  68%). 

Confusions clustered into acoustic neighborhoods for BirdNET's non-biotic results. For instance, engine was frequently misclassified as wind (31%) and vice versa (12%), generator and power tools were mutually confused (37% and 21%), and traffic was frequently misclassified as device static (27%) (**Figure 2**). 

:::{figure} ../figs/confusion_matrices.png
:label: fig-confusion
:align: center
Row-normalised confusion matrices for the multiclass classifiers (N = 80 training clips per class, counts summed across 10 random seeds). Top row: anuran group (4 classes); bottom row: non-biotic group (6 classes). Left column: BirdNET; right column: Perch. Cell values show recall percentage; raw counts in parentheses.
:::

```{raw} latex
\noindent
```
**Comparison between binary and multiclass classifiers.** For the anuran group, the mean binary − multiclass OVR AUC gap was +0.034 for BirdNET and +0.001 for Perch, indicating that Perch perfomed more consistently between binary and multiclass evaluation (**Figure 3a**). Both models showed negligible degradation under multiclass evaluation.

:::{figure} ../figs/binary_vs_multiclass_auc.png
:label: fig-binary_vs_multiclass
:align: center
Binary and multiclass one-vs-rest AUC at N = 80 training clips per class. Each point is one class–model combination; circles = BirdNET, squares = Perch. The dashed diagonal marks binary = multiclass AUC; points below it indicate that binary evaluation overstated performance. Left panel: anuran group (4 classes). Right panel: non-biotic group (6 classes).
:::

However, for the non-biotic group, BirdNET's mean gap was +0.122 while Perch's mean gap was smaller at −0.016 (negative values indicate that multiclass performed slightly better than binary) (**Figure 3b**). The largest gaps in performance were between BirdNET’s binary and multiclass classifiers for traffic (+0.188) and power tools (+0.158).
The architecture-dependent gap was specific to the non-biotic group (**Figure 3**). The binary AUC alone did not predict the relative rank of BirdNET and Perch under multiclass evaluation.

```{raw} latex
\noindent
```
**Embedding-space geometry.** In Perch’s t-SNE space, yellow-legged frog formed an isolated cluster, consistent with 100% recall. Woodhouse’s toad formed a separable subcluster, while PCF and bullfrog overlapped (**Figure 4a**).

In the non-biotic group, power tools and generator formed distinct clusters, while engine, traffic, device static, and wind intermixed slightly in a central region, visually consistent with their multiclass confusion patterns (**Figure 4b**). 

:::{figure} ../figs/tsne_embeddings.png
:label: fig-tsne
:align: center
t-SNE projection of test-set embeddings (50 clips per class) for BirdNET (left column) and Perch (right column). Top row: anuran group; bottom row: non-biotic group. Each point is one audio clip; colours denote class; stars mark class centroids. Cluster overlap corresponds to off-diagonal confusion in the multiclass classifier.
:::

### Discussion

Binary AUC overstates non-avian classifier performance in a deployment-relevant sense. The discrepancy is architecture- and class-dependent: BirdNET's binary AUC overstated non-biotic discriminability by an average of 0.12 AUC units, while Perch's binary AUC closely tracked its multiclass performance.

Model rankings change between evaluation paradigms. BirdNET and Perch were comparable under binary AUC but Perch was substantially more robust under multiclass evaluation, particularly for non-biotic sounds. This suggests that binary AUC, the dominant metric in PAM model benchmarking, may select suboptimal models for real-world multi-class deployment.

Ghani et al. (2023) established that bird-pretrained embeddings (BirdNET, Perch) outperform AudioSet-pretrained embeddings for non-bird bioacoustic transfer. Our results extend this to anthropogenic and abiotic sound classes that are acoustically distant from bird vocalizations. We additionally demonstrate that binary AUC obscures architecture-quality differences within bird-pretrained models themselves. The Perch advantage over BirdNET is invisible under binary evaluation, but substantial under multiclass evaluation.

The non-biotic mechanical cluster (engine, traffic, device static, wind) represents acoustically similar broadband non-stationary sources. Distinguishing them in multiclass deployment is intrinsically difficult and likely requires either model improvements or hierarchical workflows. Leakage between sound classes like engine and traffic or device static and wind, which often sound similar, may have led to partial confusion between classes. Likewise, distinguishing broadband frequency features in a pre-trained avian model, with a short context window of 3 to 5 seconds, is likely limiting the potential of models to derive sufficient context to make an accurate assignment. 

Conversely, power tools and generator formed distinct clusters and were more reliably classified. This is possibly because power tools can be used in shorter bursts and have distinct harmonic characteristics. Similarly, while generators do exhibit a broadband frequency range, they do have a reliable stationarity of sound, possibly allowing the model to perform well even within a small context window.

Some confusion patterns reflect annotation contamination rather than inherent acoustic similarity. Pacific chorus frog and arroyo/red-spotted toad were confirmed in each other's positive clips (approximately [15]% of PCF positives and [15]% of toad positives contained the other species in the background), partly accounting for their BirdNET multiclass confusion.

Engine and traffic likely have similar cross-contamination because they co-occur at sites. Device static annotations were heterogeneous in source, contributing to its variable confusion patterns. These quality issues are likely representative of the field rather than unique to this dataset. Most PAM workflows tightly annotate from raw recordings containing overlapping sounds. Importantly, the resulting label noise appears to affect multiclass evaluation more visibly than binary evaluation.

Call context may also have influenced the lack of confusion as well. Yellow-legged frog calls are underwater, and recordings were collected from ARUs submerged in streams, whereas other anuran calls were collected on above stream ARUs. Thus, the yellow-legged frog 100% recall may be an artifact of deployment-context features (stream background acoustics, recording equipment, site-specific noise) rather than call features alone. We cannot disentangle call-feature recognition from context-feature recognition without cross-site evaluation, which our dataset does not support. This represents a known and underreported failure mode for PAM model evaluation (CITE).

For non-avian PAM tasks involving discrimination among multiple acoustically related classes (e.g., disturbance source attribution, sympatric species classification), Perch should be preferred over BirdNET. We encourage practicioners to report multiclass evaluation within ecologically coherent class groups, not only per-class binary AUC, since binary AUC underestimates deployment issues and obscures relevant model differences. We suggest that researchers, where feasible, tightly annotate calls with isolated background, rather than raw clips with mixed sources, to minimize cross-class contamination in training data. Finally, practicioners should use hierarchical classification workflows, first checking biotic vs. non-biotic and then within-group classification, whenever the deployment target is acoustically confusable with environmental noise.

We constrained this study to California, hence generalization to other ecosystems remains untested. As we used limited classes per group (4 anuran, 6 non-biotic), finer or broader-grained groupings may reveal additional patterns. Annotation contamination was confirmed for some classes it was not systematically quantified. While this wouldn’t affect the gap we found from binary to multiclass, it may lead to improved performance in our multiclass classifiers. Yellow-legged frog results may partly reflect deployment-context features, and cross-site evaluation would be needed to resolve this. Finally, we have further confirmed that image-pretrained CNN baselines (ResNet, MobileNet, VGG) are weaker comparators than the bird pretrained baselines. However, comparison to AudioMAE, AST, or VGGish would strengthen architecture-pretraining inference. 

We conclude that cross-site generalization tests are necessary to disentangle call features from deployment-context features. Multiclass evaluation could be extended to additional ecologically coherent groups (e.g., bat species, marine mammals, multi-source disturbance taxonomies). Direct comparison could be conducted with audio-pretrained transformers (AST, AudioMAE) under the multiclass framework introduced here.

### References

Ghani, B., Denton, T., Kahl, S., & Klinck, H. (2023). Global birdsong embeddings enable superior transfer learning for bioacoustic classification. Scientific Reports, 13, 22876. https://doi.org/10.1038/s41598-023-49989-z

Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. Ecological Informatics, 61, 101236. https://doi.org/10.1016/j.ecoinf.2021.101236

Stowell, D. (2022). Computational bioacoustics with deep learning: a review and roadmap. PeerJ, 10, e13152. https://doi.org/10.7717/peerj.13152

Sugai, L. S. M., Silva, T. S. F., Ribeiro, J. W., & Llusia, D. (2019). Terrestrial passive acoustic monitoring: review and perspectives. BioScience, 69(1), 15–25. https://doi.org/10.1093/biosci/biy147

van Merriënboer, B., Dumoulin, V., Hamer, J., Harrell, L., Burns, A., & Denton, T. (2025). Perch 2.0: The Bittern Lesson for Bioacoustics. arXiv preprint arXiv:2508.04665. https://doi.org/10.48550/arXiv.2508.04665

```{raw} latex
\newpage
```

### Supplemental Material

**Additional binary classifiers for CDFW.** Beyond the 10 classes in the benchmarking experiment, we trained binary classifiers for an additional 13 sound classes: four that were not assignable to a coherent multiclass group (human vocal, coyote, field cricket, nutria), and nine with insufficient labeled data for the benchmarking experiment (metal, airplane, thunder, gun, fireworks, dog, water, human non-vocal, branches; 60 or fewer positive clips). For these classifiers, we used BirdNET as the frozen feature extractor based on its strong performance across class types in the benchmarking experiment. All audio clips were standardized to 3 seconds per BirdNET's input requirements. We trained each binary classifier on an 80-20 split of available positive clips, with negatives sampled evenly from the positive pools of the other 22 classes. As in the benchmarking experiment, we report performance as area under the receiver operating characteristic curve (AUC-ROC), where 1.0 indicates perfect ranking of positives over negatives and 0.5 indicates random performance. Trained models for all 23 classes are included in the accompanying data folder.

:::{table} Additional nine sound classes not included in the experiment.
:label: tbl-additional-classes

| Sound class | Positive clips | Negative clips | Audio clip origin |
|---|---|---|---|
| Human vocal | 117 | 117 | 40 | Margin sampling, opportunistic |
| Coyote | 101 | 101 | 40 | Margin sampling, Mojave |
| Field cricket | 605 | 605 | 160 | Margin sampling, Mojave |
| Nutria | 245 | 245 | 160 | Judas Nutria team, CDFW |
| Metal clanging | 60 | 60 | Opportunistic |
| Airplane | 52 | 52 | Opportunistic |
| Thunder | 21 | 21 | Opportunistic |
| Gun | 19 | 19 | Margin sampling, all CDFW sites |
| Fireworks | 13 | 13 | Margin sampling, all CDFW sites |
| Dog | 12 | 12 | Opportunistic |
| Water | 11 | 11 | ARU location info, stream likely; opportunistic |
| Human non-vocal | 8 | 8 | Margin sampling, all CDFW sites |
| Branches | 3 | 3 | Opportunistic |
:::

### Additional potential references
PAM and soundscape ecology background

Sugai, L. S. M., Silva, T. S. F., Ribeiro, J. W., & Llusia, D. (2019). Terrestrial passive acoustic monitoring: review and perspectives. BioScience, 69(1), 15–25.
Stowell, D. (2022). Computational bioacoustics with deep learning: a review and roadmap. PeerJ, 10, e13152.
Pijanowski, B. C., Villanueva-Rivera, L. J., Dumyahn, S. L., Farina, A., Krause, B. L., Napoletano, B. M., Gage, S. H., & Pieretti, N. (2011). Soundscape ecology: the science of sound in the landscape. BioScience, 61(3), 203–216.

Bioacoustic benchmark and evaluation methodology

Knight, E. C., Hannah, K. C., Foley, G. J., Scott, C. D., Brigham, R. M., & Bayne, E. (2017). Recommendations for acoustic recognizer performance assessment with application to five common automated signal recognition programs. Avian Conservation and Ecology, 12(2), 14.

Non-avian PAM applications

Buxton, R. T., McKenna, M. F., Mennitt, D., Fristrup, K., Crooks, K., Angeloni, L., & Wittemyer, G. (2017). Noise pollution is pervasive in U.S. protected areas. Science, 356(6337), 531–533.
Crump, P. F., & Houlahan, J. (2017). Designing better frog call recognition models. Ecology and Evolution, 7(9), 3087–3099.

Transfer learning in bioacoustics

Dufourq, E., Batist, C., Foquet, R., & Durbach, I. (2022). Passive acoustic monitoring of animal populations with transfer learning. Ecological Informatics, 70, 101688.
Lauha, P., Somervuo, P., Lehikoinen, P., Geres, L., Richter, T., Seibold, S., & Ovaskainen, O. (2022). Domain-specific neural networks improve automated bird sound recognition already with small amount of local data. Methods in Ecology and Evolution, 13(12), 2799–2810.

Anuran-specific PAM

Acevedo, M. A., Corrada-Bravo, C. J., Corrada-Bravo, H., Villanueva-Rivera, L. J., & Aide, T. M. (2009). Automated classification of bird and amphibian calls using machine learning: A comparison of methods. Ecological Informatics, 4(4), 206–214.

+++ {"part": "competing_interests"}
The author declares no competing interests.
+++

