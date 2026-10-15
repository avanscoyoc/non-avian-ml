---
title: Transferring pretrained audio models to artificial noise and abiotic sound
subtitle: Experiment brief
short_title: Pretrained audio models for noise detection
authors:
  - name: Amy Van Scoyoc
    email: avanscoyoc@berkeley.edu
    orcid: 0000-0001-8638-935X
    website: https://github.com/avanscoyoc/non-avian-ml
abstract: |
  Autonomous recording units (ARUs) capture artificial noise and abiotic sound alongside wildlife, and pretrained audio models are the obvious starting point for detecting them. We tested how well six frozen pretrained models transfer to 14 noise classes by training a small classifier head on each model's embeddings. The models were four bird models, a general-audio model and an image model. All test clips came from recordings never used for training. About 40 labelled clips per class brought most classes close to their best performance. The sound class mattered more than the model: wind and engine were easy for every model, and generator and power tools were hard for every model. A trained head usually matched or beat a model's own built-in label for the class. Errors fell between acoustically similar sounds in binary detectors, in a multiclass head and in the embeddings themselves. Longer input windows helped only generator.
data_availability: |
  Labeled audio clips are archived on Zenodo at [doi:10.5281/zenodo.20534256](https://doi.org/10.5281/zenodo.20534256).
keywords:
  - passive acoustic monitoring
  - transfer learning
  - anthropogenic noise
  - bioacoustics
  - BirdNET
  - Perch
exports:
  - format: pdf
    template: ./lapreprint-custom
    output: brief.pdf
    venue_footer: biorxiv
---

```{raw} latex
\captionsetup{font=small, skip=8pt}
% Multi-panel figures use the full page width (text column plus the left margin).
\let\briefoldfigure\figure
\let\briefoldendfigure\endfigure
\renewenvironment{figure}[1][htbp]{\briefoldfigure[#1]\begin{fullwidth}}{\end{fullwidth}\briefoldendfigure}
```

## Main findings

1. **About 40 labelled clips per class is enough for most classes.** A small head trained on frozen embeddings
   reaches a mean test AUC of 0.69–0.97 across the well-sampled classes at N = 40 clips per class, and most curves
   are close to their best by then.
2. **The sound class matters more than the model.** Wind and engine are easy for every model, and generator and
   power tools are hard for every model. BirdNET 2.4 is best on average. An ImageNet image model does about as well
   as the general-audio model CNN14. Where a class's training clips come from only two recordings (power tools,
   human vocal), adding more clips does not help.
3. **Training a head usually matches or beats a model's built-in label.** For classes a model already has a label
   for, the trained head clears the label in 13 of 29 cases, typically by N = 5 clips, and ties near the ceiling in 7.
4. **Errors fall between acoustically similar sounds:** vehicle noise, impulsive sounds, wind and thunder, and device
   static and wind. The same confusions appear in binary detectors, in a multiclass head and in the embeddings
   themselves. Binary and multiclass heads discriminate each class about equally well.
5. **Longer input windows help only generator.** With BirdNET 3.0, windows of 5–10 s instead of 3 s improve
   generator and leave the other classes unchanged.

## Approach

**Data and split.** Clips are 3 s long and come from ARU recordings (**Table 1**). The 14 target classes comprise 11
artificial and 3 abiotic sound classes. Nine further classes (frogs and toads, field cricket, coyote, nutria, dog,
branches) serve only as negatives, so that detectors face realistic field sounds. Each source recording is assigned
to either the test or the training set, never both, so every test clip comes from a recording the classifier never
saw. Each class has 50 test clips where the data allow, and the same test clips are used in every experiment. Seven
low-sample classes (fewer than 90 clips) reach only N = 5–20 and have test sets of 3–30 clips. Their results are
provisional and are reported in the supplement.

**Models.** Six pretrained encoders are used frozen (**Table S1**):
- four bird models: BirdNET 2.4, BirdNET 3.0, Perch v8 and Perch 2.0;
- a general-audio model: CNN14, trained on AudioSet;
- an image model as a non-audio baseline: ResNet-18, trained on ImageNet.

**Adaptation.** Each clip is embedded once per model. For each class, a small multilayer perceptron (MLP) head with
two hidden layers (256 and 64 units) is trained on N clips of the class and N negatives drawn equally from the other
classes, for N = 5, 10, 20, 40, 80 and 160 as far as the data allow. Each N is repeated with 10 random training
draws, and results are means with 95% confidence intervals across draws. Heads follow the same schedule at every N
(20 epochs, batch size 32), so smaller training sets receive fewer gradient steps. A linear probe fitted to
convergence gives the same curves (**Figure S2**). N = 40 is the headline training size.

**Zero-shot baseline.** Some models already have an output label for a class, such as BirdNET 2.4's “Engine” or the
FSD50K and AudioSet classes of Perch 2.0 and CNN14. For those, the label's own score is evaluated directly as N = 0
(**Table S2**). We call such classes “known” to the model and the rest “new”.

**Metric.** Performance is the area under the ROC curve on the test clips (AUC; 0.5 = chance, 1 = perfect
separation). Cross-detection and confusion matrices report proportions of test clips.

## Data

:::{table} **Clips and source recordings per target class.** Each source recording belongs to either the test or the training set, never both. Max N is the largest training size a class supports (N clips of the class plus N negatives). Asterisks mark low-sample classes (fewer than 90 clips), for which about half of the recordings are held out for testing; their results are provisional. Nine further classes (3,256 clips from 311 recordings) serve only as negatives. In total there are 6,547 clips from 788 recordings, 163 of which are test recordings.
:label: tbl-data

| Class | Type | Recordings / clips | Test recordings / clips | Training recordings / clips | Max N |
|---|---|---|---|---|---|
| Device static | artificial | 168 / 1,416 | 9 / 50 | 159 / 1,366 | 160 |
| Generator | artificial | 32 / 437 | 10 / 50 | 22 / 384 | 160 |
| Traffic | artificial | 53 / 297 | 13 / 50 | 40 / 242 | 160 |
| Engine | artificial | 115 / 170 | 38 / 50 | 76 / 117 | 80 |
| Power tools | artificial | 11 / 162 | 9 / 29 | 2 / 133 | 80 |
| Human vocal | artificial | 16 / 117 | 14 / 50 | 2 / 64 | 40 |
| Metal clanging \* | artificial | 28 / 60 | 13 / 30 | 15 / 30 | 20 |
| Airplane \* | artificial | 12 / 52 | 6 / 25 | 6 / 27 | 20 |
| Gun \* | artificial | 7 / 19 | 5 / 9 | 2 / 10 | 10 |
| Fireworks \* | artificial | 4 / 13 | 2 / 6 | 2 / 7 | 5 |
| Human non-vocal \* | artificial | 2 / 8 | 1 / 3 | 1 / 5 | 5 |
| Wind | abiotic | 106 / 508 | 11 / 50 | 94 / 453 | 160 |
| Thunder \* | abiotic | 5 / 21 | 2 / 8 | 3 / 13 | 10 |
| Water \* | abiotic | 3 / 11 | 2 / 5 | 1 / 6 | 5 |
:::

Power tools and human vocal have many clips, but only two training recordings each. Their training clips are
therefore less independent than the counts suggest.

## How many clips are needed?

Classes differed more than models (**Figures 1 and 2**). At N = 40, mean AUC across models ranged from 0.97 for wind
and 0.90 for engine down to 0.73 for generator and 0.69 for power tools. Mean AUC by model over the well-sampled
classes ranged only from 0.78 to 0.87:

| Model | Mean AUC at N = 40 |
|---|---|
| BirdNET 2.4 | 0.87 |
| Perch 2.0 | 0.85 |
| Perch v8 | 0.83 |
| CNN14 | 0.81 |
| ResNet-18 | 0.80 |
| BirdNET 3.0 | 0.78 |

The median curve came within 0.02 AUC of its best by N = 40. Device static and traffic were still rising at N = 160.
Doubling from 40 to 80 clips added 0.01–0.04 AUC, depending on the model.

In 29 class–model pairs the model has a built-in label. Comparing the trained head's 95% confidence interval with
the label's zero-shot AUC:
- **13 pairs:** the head clearly beat the label, with a median break-even at N = 5.
- **7 pairs:** both were near the ceiling (AUC of 0.97 or more).
- **3 pairs:** the intervals overlapped.
- **6 pairs:** the built-in label stayed better. These were power tools and human vocal, whose training clips each
  come from only two recordings, and the low-sample classes thunder and water.

No model has a label for generator.

:::{figure} ../figs/saturation_main.pdf
:label: fig-saturation
:width: 82%
**Test performance rises with training size and levels off near N = 40 for most well-sampled classes.** Test AUC of binary detectors (an MLP head trained on frozen embeddings) against the number of training clips per class, N (log scale), for the seven well-sampled classes and six pretrained models. Lines show means over 10 random training draws, and shaded bands show 95% confidence intervals. Open markers at N = 0 show the zero-shot AUC of a model's built-in label for the class, where one exists; panel subtitles name those models. The dotted line marks chance. Each class has 50 test clips (power tools, 29) and an equal number of negatives.
:::

:::{figure} ../figs/saturation_summary.pdf
:label: fig-heatmap
:width: 72%
**Every model struggles with the same classes, and a trained head usually matches or beats a built-in label.** Mean test AUC at N = 40 for each class (rows) and model (columns), over 10 random training draws. Low-sample classes (marked \*, below the horizontal line) are shown at the largest training size they support (\@N). Outlined cells mark classes for which the model has a built-in label; “zs” gives that label's zero-shot AUC.
:::

## What do binary detectors respond to?

Each binary detector is trained against a broad background. Its 40 negatives are drawn from 22 other classes, so
any one noise class contributes only 0–6 clips per draw. We applied each detector to the test clips of every class
at a cutoff of 0.5 (**Figure 3**). Depending on the model, the well-sampled detectors had a mean recall of 0.55–0.62
and a mean false-alarm rate on other noise classes of 0.14–0.23. False alarms fell into the same three groups for
every model:
- impulsive sounds: metal clanging, gun and fireworks;
- wind and thunder;
- vehicle noise: traffic, engine and airplane.

Training each detector on all of its class's training clips mainly lowered false alarms, to 0.11–0.17
(**Figure S3**). Recall rose for device static (from 0.53–0.73 to 0.71–0.85) and for traffic. It did not rise for
power tools or human vocal, whose training clips come from only two recordings each. Tuning a separate cutoff for
each detector on its training data did not improve the balance between recall and false alarms (Supplementary note).

:::{figure} ../figs/cross_detection.pdf
:label: fig-cross
:width: 100%
**Binary detectors respond to acoustically similar noise classes.** Each class's binary detector (rows) was applied to the test clips of every class (columns; number of test clips in brackets). Detectors were trained on N = 40 clips per class, or on the largest N a low-sample class supports (shown in the row label). Cells give the proportion of a class's test clips that the detector labels positive at a probability cutoff of 0.5, averaged over 10 random training draws. The outlined diagonal is recall, and off-diagonal cells are false-alarm rates. Detectors act independently, so rows need not sum to one.
:::

## One multiclass classifier

The alternative to separate detectors is one head that names the noise. A single 7-class head was trained on the
same 40 clips per class as the binary detectors, using the seven classes with at least 40 training clips
(**Figure 4**):
- It recognised wind reliably (recall 0.90–0.97).
- It assigned most generator and power tools clips to other classes (recall 0.13–0.32), chiefly traffic and engine.
- It often labelled device static as wind (0.19–0.33).

Macro F1 ranged from 0.43 (ResNet-18) to 0.55 (Perch 2.0).

:::{figure} ../figs/confusion_matrices.pdf
:label: fig-confusion
:width: 100%
**The multiclass head recognises wind reliably but spreads generator and power tools across other noise classes.** Row-normalised confusion matrices for a 7-class MLP head per model. Each head was trained on the same 40 clips per class as the binary detectors and evaluated on the same test clips (number of test clips per true class in brackets). Counts are pooled over 10 random training draws, and the outlined diagonal is recall. Only classes with at least 40 training clips are included.
:::

## Embedding geometry

The same structure is visible before any classifier is trained (**Figure 5**). Device static forms a compact group
in most models, whereas generator, traffic, engine and power tools overlap in every model's embedding.

:::{figure} ../figs/tsne_embeddings.pdf
:label: fig-tsne
:width: 100%
**Device static forms a compact group in most embeddings, while the mechanical noise classes overlap.** Two-dimensional t-SNE projections (perplexity 30) of the frozen embeddings of the 329 test clips in the multiclass experiment, one panel per model. Each point is a clip, coloured by class, and stars mark class centroids in the projection. No classifier is involved. t-SNE preserves local neighbourhoods but not global distances, so only the grouping should be interpreted.
:::

## Binary detectors or one multiclass head?

Binary detectors and the multiclass head discriminated each class about equally well (**Figure 6**). Across all
class–model pairs, the multiclass one-vs-rest AUC differed from the binary AUC by −0.002 on average (range −0.14 to
+0.10). The largest drops were for device static (down to −0.14). Wind is a hard negative for device static, and it
makes up the multiclass negatives but only a small share of the binary ones.

:::{figure} ../figs/binary_vs_multiclass_auc.pdf
:label: fig-binary-multiclass
:width: 85%
**Binary detectors and the multiclass head discriminate each class about equally well.** Binary detector AUC at N = 40 (x-axis) against the multiclass head's one-vs-rest AUC (y-axis), each a mean over 10 random training draws. Each point is one class, and the dashed line marks equal performance. Both classifiers were trained on the same 40 clips per class and evaluated on the same positive test clips. Binary negatives are drawn from all other classes, including animal sounds, whereas multiclass negatives are the test clips of the other six noise classes.
:::

## Input window length

BirdNET 3.0 accepts inputs of any length. We cut the same events at 3, 5 and 10 s (**Table S3**) to test whether
long, steady broadband noise benefits from more context than short pulsed sounds. Longer windows helped only
generator, whose AUC at N = 20 rose from 0.84 at 3 s to 0.89 at 5 and 10 s (**Figure 7**). The other classes
showed no gain:
- device static (0.93) and wind (0.88–0.89) did not change;
- traffic stayed low (0.62–0.68) with no consistent trend;
- the pulsed animal sounds stayed at 0.98–0.99.

:::{figure} ../figs/window_size.pdf
:label: fig-window
:width: 100%
**Longer input windows improve BirdNET 3.0 only for generator, and pulsed animal sounds are near ceiling at every length.** Test AUC of BirdNET 3.0 binary detectors for 3, 5 and 10 s input windows centred on the same acoustic events, at N = 5, 10 and 20 training clips per class. The top row shows broadband noise classes and the bottom row pulsed animal sounds, for contrast; the right-hand panels average each group. Each class has 25 test clips from held-out recordings plus 25 negatives, shared across window lengths so that comparisons are paired. Lines show means over 10 random training draws, and bands show 95% confidence intervals. The dotted line marks chance.
:::

```{raw} latex
\clearpage
\setcounter{figure}{0}\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{table}{0}\renewcommand{\thetable}{S\arabic{table}}
```

## Supplementary material

:::{table} **Pretrained models.** Embedding size, input format and the built-in labels used for zero-shot evaluation. The Perch models receive the 3 s clip centre-padded to 5 s.
:label: tbl-models

| Model | Embedding | Input | Built-in labels used for zero-shot |
|---|---|---|---|
| BirdNET 2.4 | 1024-D | 3 s at 48 kHz | Engine, Power tools, Human vocal, Human non-vocal, Gun, Fireworks |
| BirdNET 3.0 | 1280-D | 3 s at 32 kHz (3, 5 and 10 s in Figure 7) | Human vocal |
| Perch v8 | 1280-D | 5 s at 32 kHz | none (bird labels only) |
| Perch 2.0 | 1536-D | 5 s at 32 kHz | FSD50K classes (Table S2) |
| CNN14 (AudioSet) | 2048-D | 3 s at 32 kHz | AudioSet classes (Table S2) |
| ResNet-18 (ImageNet) | 512-D | 64-band mel spectrogram, 3 s at 16 kHz | none |
:::

:::{table} **Built-in labels used for zero-shot evaluation.** Only labels that directly name the class are used, and a clip's zero-shot score is the maximum over the listed labels. Perch v8 and ResNet-18 have no such labels, and no model has a label for generator.
:label: tbl-labels

| Class | BirdNET 2.4 | BirdNET 3.0 | Perch 2.0 | CNN14 |
|---|---|---|---|---|
| Device static | | | | Static |
| Engine | Engine | | Engine; Idling | Engine; Idling |
| Traffic | | | Traffic noise and roadway noise | Traffic noise, roadway noise |
| Power tools | Power tools | | Power tool; Drill | Power tool; Drill |
| Human vocal | Human vocal | Human vocal | Speech; Human voice; Homo sapiens | Speech |
| Metal clanging | | | | Clang |
| Airplane | | | Fixed-wing aircraft and airplane | Fixed-wing aircraft, airplane |
| Gun | Gun | | Gunshot and gunfire | Gunshot, gunfire |
| Fireworks | Fireworks | | Fireworks | Fireworks |
| Human non-vocal | Human non-vocal | | | |
| Wind | | | Wind | Wind; Wind noise (microphone) |
| Thunder | | | Thunder | Thunder |
| Water | | | Water | Water |
:::

:::{table} **Clips available for the window-length experiment.** Number of clips (source recordings) cut at all three window lengths. Negatives for each class are drawn from the other classes in this set.
:label: tbl-windows

| Class | Group | Clips (recordings) |
|---|---|---|
| Device static | broadband | 239 (147) |
| Generator | broadband | 79 (28) |
| Wind | broadband | 79 (79) |
| Traffic | broadband | 48 (36) |
| American bullfrog | pulsed | 378 (13) |
| Field cricket | pulsed | 539 (103) |
| Pacific chorus frog | pulsed | 705 (41) |
:::

:::{figure} ../figs/saturation_low_sample.pdf
:label: fig-low-sample
:width: 82%
**Saturation curves for the low-sample classes (provisional).** As Figure 1, for the seven classes with fewer than 90 clips. Curves end at the largest training size each class supports (N = 5–20). Test sets contain 3–30 clips from 1–13 recordings. The confidence intervals reflect only variation among training draws, so they understate the uncertainty that comes from such small test sets.
:::

:::{figure} ../figs/head_comparison.pdf
:label: fig-head
:width: 100%
**A linear probe reproduces the saturation curves of the MLP head.** Mean test AUC over the seven well-sampled classes for the MLP head used throughout (solid) and a linear probe (standardised logistic regression fitted to convergence; dashed). Both were trained on the same embeddings and training draws and evaluated on the same test clips. The MLP follows a fixed 20-epoch schedule, so smaller training sets receive fewer gradient steps. The linear probe is fitted to convergence at every N and shows the same rise with N, so the saturation curves are not an artefact of that schedule.
:::

:::{figure} ../figs/cross_detection_all_data.pdf
:label: fig-cross-all
:width: 100%
**Training on all available clips mainly reduces false alarms.** As Figure 3, but each detector is trained on all training clips of its class (N in row labels). The test recordings remain held out.
:::

**Supplementary note: detection cutoff.** The cross-detection figures use a probability cutoff of 0.5, the natural
decision threshold for a head trained on equal numbers of positives and negatives. As a check, each detector was
also given its own cutoff, chosen to maximise F1 on out-of-fold predictions from its training data, with folds
split by recording. Tuned cutoffs raised recall and false alarms together without consistently improving their
balance. For low-sample detectors, whose folds held only 5–10 training clips, the tuned cutoffs fell to about
0.2–0.3. They were therefore not used.
