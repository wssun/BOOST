# Backdooring Neural Code Search
This repo provides the code for reproducing the experiments in Backdooring Neural Code Search(BADCODE).

## An Overview to BADCODE
![framework](figures/framework.png)

## Figure of Data (including perturbed, poisoned)
![framework](figures/data.png)

## Glance
```
├─── data
│    ├─── cifar
│    ├─── sunrise
│    ├─── tirgger
│    │    ├─── other
│    │    ├─── refool
│    │    ├─── wanet
├─── figures
│    ├─── framework.png
│    ├─── data.png
├─── models
├─── src
│    ├─── CodeBERT
│    │    ├─── evaluate_attack
│    │    │    ├─── evaluate_attack.py
│    │    │    ├─── mrr_poisoned_model.py
│    │    ├─── mrr.py
│    │    ├─── run_classifier.py
│    │    ├─── utils.py
│    ├─── CodeT5
│    │    ├─── evaluate_attack
│    │    │    ├─── evaluate_attack.py
│    │    │    ├─── mrr_poisoned_model.py
│    │    ├─── _utils.py
│    │    ├─── configs.py
│    │    ├─── models.py
│    │    ├─── run_search.py
│    │    ├─── utils.py
│    ├─── stealthiness
│    │    ├─── defense
│    │    │    ├───activation_clustering.py
│    │    │    ├───spectral_signature.py
├─── utils
│    ├─── results
│    │    ├─── matching_pair
│    │    ├─── selecting_trigger
│    ├─── vocab_frequency.py
│    ├─── select_trigger.py
├─── README.md
├─── trigger-injected samples.pdf
```

## Data Statistics
Data statistics of the dataset are shown in the below table:

|       | Python  |  Java   |
| ----- |:-------:|:-------:|
| Train | 412,178 | 454,451 |
| Valid | 23,107  | 15,328  |
| Test  | 22,176  | 26,909  |

## Backdoor attack
