| class | model | label_status | zero_shot_auc | auc_at_40 | ci_40 | best_auc | best_n | plateau_n | break_even_n | outcome | low_sample | train_recordings | train_clips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Device static | BirdNET 2.4 | new |  | 0.825 | 0.04 | 0.906 | 160 | 160 |  |  | False | 159 | 1366 |
| Device static | BirdNET 3.0 | new |  | 0.768 | 0.033 | 0.862 | 160 | 160 |  |  | False | 159 | 1366 |
| Device static | Perch v8 | new |  | 0.859 | 0.032 | 0.88 | 80 | 80 |  |  | False | 159 | 1366 |
| Device static | Perch 2.0 | new |  | 0.886 | 0.029 | 0.929 | 160 | 160 |  |  | False | 159 | 1366 |
| Device static | CNN14 (AudioSet) | known | 0.711 | 0.726 | 0.041 | 0.886 | 160 | 160 | 80.0 | adaptation beats label | False | 159 | 1366 |
| Device static | ResNet-18 (ImageNet) | new |  | 0.84 | 0.047 | 0.918 | 160 | 80 |  |  | False | 159 | 1366 |
| Generator | BirdNET 2.4 | new |  | 0.79 | 0.02 | 0.791 | 160 | 20 |  |  | False | 22 | 384 |
| Generator | BirdNET 3.0 | new |  | 0.739 | 0.015 | 0.785 | 160 | 80 |  |  | False | 22 | 384 |
| Generator | Perch v8 | new |  | 0.74 | 0.012 | 0.742 | 160 | 20 |  |  | False | 22 | 384 |
| Generator | Perch 2.0 | new |  | 0.771 | 0.028 | 0.788 | 160 | 40 |  |  | False | 22 | 384 |
| Generator | CNN14 (AudioSet) | new |  | 0.645 | 0.03 | 0.734 | 160 | 80 |  |  | False | 22 | 384 |
| Generator | ResNet-18 (ImageNet) | new |  | 0.688 | 0.028 | 0.813 | 160 | 160 |  |  | False | 22 | 384 |
| Traffic | BirdNET 2.4 | new |  | 0.788 | 0.032 | 0.896 | 160 | 160 |  |  | False | 40 | 242 |
| Traffic | BirdNET 3.0 | new |  | 0.773 | 0.016 | 0.84 | 160 | 160 |  |  | False | 40 | 242 |
| Traffic | Perch v8 | new |  | 0.778 | 0.021 | 0.835 | 160 | 80 |  |  | False | 40 | 242 |
| Traffic | Perch 2.0 | known | 0.44 | 0.806 | 0.024 | 0.836 | 160 | 80 | 5.0 | adaptation beats label | False | 40 | 242 |
| Traffic | CNN14 (AudioSet) | known | 0.692 | 0.81 | 0.02 | 0.87 | 160 | 160 | 20.0 | adaptation beats label | False | 40 | 242 |
| Traffic | ResNet-18 (ImageNet) | new |  | 0.77 | 0.014 | 0.848 | 160 | 160 |  |  | False | 40 | 242 |
| Engine | BirdNET 2.4 | known | 0.888 | 0.952 | 0.012 | 0.952 | 40 | 20 | 20.0 | adaptation beats label | False | 76 | 117 |
| Engine | BirdNET 3.0 | new |  | 0.847 | 0.017 | 0.847 | 40 | 40 |  |  | False | 76 | 117 |
| Engine | Perch v8 | new |  | 0.867 | 0.02 | 0.89 | 80 | 80 |  |  | False | 76 | 117 |
| Engine | Perch 2.0 | known | 0.698 | 0.903 | 0.017 | 0.91 | 80 | 20 | 5.0 | adaptation beats label | False | 76 | 117 |
| Engine | CNN14 (AudioSet) | known | 0.852 | 0.932 | 0.007 | 0.94 | 80 | 40 | 20.0 | adaptation beats label | False | 76 | 117 |
| Engine | ResNet-18 (ImageNet) | new |  | 0.88 | 0.008 | 0.9 | 80 | 40 |  |  | False | 76 | 117 |
| Power tools | BirdNET 2.4 | known | 0.929 | 0.82 | 0.009 | 0.839 | 80 | 40 |  | label better | False | 2 | 133 |
| Power tools | BirdNET 3.0 | new |  | 0.654 | 0.024 | 0.691 | 80 | 80 |  |  | False | 2 | 133 |
| Power tools | Perch v8 | new |  | 0.638 | 0.028 | 0.659 | 5 | 5 |  |  | False | 2 | 133 |
| Power tools | Perch 2.0 | known | 0.715 | 0.702 | 0.017 | 0.716 | 80 | 5 |  | adaptation >= label (CI overlaps) | False | 2 | 133 |
| Power tools | CNN14 (AudioSet) | known | 0.698 | 0.674 | 0.017 | 0.697 | 80 | 80 |  | label better | False | 2 | 133 |
| Power tools | ResNet-18 (ImageNet) | new |  | 0.647 | 0.016 | 0.647 | 40 | 5 |  |  | False | 2 | 133 |
| Human vocal | BirdNET 2.4 | known | 0.92 | 0.932 | 0.012 | 0.932 | 40 | 40 |  | adaptation >= label (CI overlaps) | False | 2 | 64 |
| Human vocal | BirdNET 3.0 | known | 0.548 | 0.738 | 0.015 | 0.738 | 40 | 40 | 5.0 | adaptation beats label | False | 2 | 64 |
| Human vocal | Perch v8 | new |  | 0.932 | 0.005 | 0.932 | 40 | 40 |  |  | False | 2 | 64 |
| Human vocal | Perch 2.0 | known | 0.837 | 0.93 | 0.005 | 0.93 | 40 | 40 | 10.0 | adaptation beats label | False | 2 | 64 |
| Human vocal | CNN14 (AudioSet) | known | 0.99 | 0.909 | 0.006 | 0.909 | 40 | 20 |  | label better | False | 2 | 64 |
| Human vocal | ResNet-18 (ImageNet) | new |  | 0.768 | 0.014 | 0.768 | 40 | 40 |  |  | False | 2 | 64 |
| Metal clanging | BirdNET 2.4 | new |  |  |  | 0.712 | 20 | 20 |  |  | True | 15 | 30 |
| Metal clanging | BirdNET 3.0 | new |  |  |  | 0.624 | 20 | 20 |  |  | True | 15 | 30 |
| Metal clanging | Perch v8 | new |  |  |  | 0.724 | 20 | 20 |  |  | True | 15 | 30 |
| Metal clanging | Perch 2.0 | new |  |  |  | 0.819 | 20 | 20 |  |  | True | 15 | 30 |
| Metal clanging | CNN14 (AudioSet) | known | 0.543 |  |  | 0.756 | 20 | 10 | 5.0 | adaptation beats label | True | 15 | 30 |
| Metal clanging | ResNet-18 (ImageNet) | new |  |  |  | 0.748 | 20 | 10 |  |  | True | 15 | 30 |
| Airplane | BirdNET 2.4 | new |  |  |  | 0.794 | 20 | 20 |  |  | True | 6 | 27 |
| Airplane | BirdNET 3.0 | new |  |  |  | 0.624 | 20 | 10 |  |  | True | 6 | 27 |
| Airplane | Perch v8 | new |  |  |  | 0.813 | 20 | 20 |  |  | True | 6 | 27 |
| Airplane | Perch 2.0 | known | 0.49 |  |  | 0.753 | 20 | 20 | 5.0 | adaptation beats label | True | 6 | 27 |
| Airplane | CNN14 (AudioSet) | known | 0.797 |  |  | 0.936 | 20 | 20 | 10.0 | adaptation beats label | True | 6 | 27 |
| Airplane | ResNet-18 (ImageNet) | new |  |  |  | 0.892 | 20 | 20 |  |  | True | 6 | 27 |
| Gun | BirdNET 2.4 | known | 0.704 |  |  | 1.0 | 10 | 5 | 5.0 | adaptation beats label | True | 2 | 10 |
| Gun | BirdNET 3.0 | new |  |  |  | 0.741 | 10 | 5 |  |  | True | 2 | 10 |
| Gun | Perch v8 | new |  |  |  | 0.957 | 10 | 10 |  |  | True | 2 | 10 |
| Gun | Perch 2.0 | known | 1.0 |  |  | 1.0 | 10 | 5 |  | tie at ceiling | True | 2 | 10 |
| Gun | CNN14 (AudioSet) | known | 1.0 |  |  | 1.0 | 5 | 5 |  | tie at ceiling | True | 2 | 10 |
| Gun | ResNet-18 (ImageNet) | new |  |  |  | 0.904 | 10 | 5 |  |  | True | 2 | 10 |
| Fireworks | BirdNET 2.4 | known | 1.0 |  |  | 1.0 | 5 | 5 |  | tie at ceiling | True | 2 | 7 |
| Fireworks | BirdNET 3.0 | new |  |  |  | 0.742 | 5 | 5 |  |  | True | 2 | 7 |
| Fireworks | Perch v8 | new |  |  |  | 0.939 | 5 | 5 |  |  | True | 2 | 7 |
| Fireworks | Perch 2.0 | known | 1.0 |  |  | 1.0 | 5 | 5 |  | tie at ceiling | True | 2 | 7 |
| Fireworks | CNN14 (AudioSet) | known | 1.0 |  |  | 1.0 | 5 | 5 |  | tie at ceiling | True | 2 | 7 |
| Fireworks | ResNet-18 (ImageNet) | new |  |  |  | 0.992 | 5 | 5 |  |  | True | 2 | 7 |
| Human non vocal | BirdNET 2.4 | known | 0.889 |  |  | 1.0 | 5 | 5 | 5.0 | adaptation beats label | True | 1 | 5 |
| Human non vocal | BirdNET 3.0 | new |  |  |  | 1.0 | 5 | 5 |  |  | True | 1 | 5 |
| Human non vocal | Perch v8 | new |  |  |  | 0.978 | 5 | 5 |  |  | True | 1 | 5 |
| Human non vocal | Perch 2.0 | new |  |  |  | 1.0 | 5 | 5 |  |  | True | 1 | 5 |
| Human non vocal | CNN14 (AudioSet) | new |  |  |  | 1.0 | 5 | 5 |  |  | True | 1 | 5 |
| Human non vocal | ResNet-18 (ImageNet) | new |  |  |  | 1.0 | 5 | 5 |  |  | True | 1 | 5 |
| Wind | BirdNET 2.4 | new |  | 0.976 | 0.002 | 0.981 | 160 | 10 |  |  | False | 94 | 453 |
| Wind | BirdNET 3.0 | new |  | 0.956 | 0.011 | 0.956 | 40 | 20 |  |  | False | 94 | 453 |
| Wind | Perch v8 | new |  | 0.968 | 0.007 | 0.979 | 160 | 40 |  |  | False | 94 | 453 |
| Wind | Perch 2.0 | known | 0.977 | 0.956 | 0.015 | 0.97 | 160 | 10 |  | tie at ceiling | False | 94 | 453 |
| Wind | CNN14 (AudioSet) | known | 0.983 | 0.982 | 0.002 | 0.982 | 20 | 5 |  | tie at ceiling | False | 94 | 453 |
| Wind | ResNet-18 (ImageNet) | new |  | 0.974 | 0.002 | 0.98 | 160 | 5 |  |  | False | 94 | 453 |
| Thunder | BirdNET 2.4 | new |  |  |  | 0.967 | 10 | 5 |  |  | True | 3 | 13 |
| Thunder | BirdNET 3.0 | new |  |  |  | 0.808 | 5 | 5 |  |  | True | 3 | 13 |
| Thunder | Perch v8 | new |  |  |  | 0.969 | 10 | 10 |  |  | True | 3 | 13 |
| Thunder | Perch 2.0 | known | 0.953 |  |  | 0.936 | 10 | 5 |  | label better | True | 3 | 13 |
| Thunder | CNN14 (AudioSet) | known | 1.0 |  |  | 0.955 | 10 | 5 |  | label better | True | 3 | 13 |
| Thunder | ResNet-18 (ImageNet) | new |  |  |  | 0.989 | 10 | 10 |  |  | True | 3 | 13 |
| Water | BirdNET 2.4 | new |  |  |  | 0.796 | 5 | 5 |  |  | True | 1 | 6 |
| Water | BirdNET 3.0 | new |  |  |  | 0.916 | 5 | 5 |  |  | True | 1 | 6 |
| Water | Perch v8 | new |  |  |  | 0.752 | 5 | 5 |  |  | True | 1 | 6 |
| Water | Perch 2.0 | known | 0.76 |  |  | 0.784 | 5 | 5 |  | adaptation >= label (CI overlaps) | True | 1 | 6 |
| Water | CNN14 (AudioSet) | known | 0.76 |  |  | 0.728 | 5 | 5 |  | label better | True | 1 | 6 |
| Water | ResNet-18 (ImageNet) | new |  |  |  | 0.564 | 5 | 5 |  |  | True | 1 | 6 |
