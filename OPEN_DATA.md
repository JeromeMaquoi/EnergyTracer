This file describes a backbone markdown document, which aims to be completed with the necessary information to make the dataset open and reusable by the research community.

---

## Dataset Overview

This dataset contains energy consumption measurements collected using the **EnergyTracer** tool. The data compares the execution of a Python script containing a specific code smell against a refactored version without the smell. It includes granular energy metrics such as CPU, GPU, and DRAM energy consumed across multiple execution iterations to reliably evaluate the energy impact of the code smell.

## Experimental Context

- **Research question**: 
- **Code smell tested**: 
- **Measurement tool**: EnergyTracer
- **Statistical parameters**: two-sided Mann-Whitney U test (α = 0.05) and Vargha-Delaney A effect size.

## Instances

| Instance       | Hardware        | OS               | Architecture |
|----------------|-----------------|------------------|--------------|
| ...            | ...             | ...              | ...          |

## Dataset Structure

```
dataset/
├── README.md
└── <instance_type>/
    └── <iteration>/
        └── <sub_instance_type>/             # optional
            ├── output/
               ├── <profiler>/               # e.g., mac, carbon
               │   └── <measure_id>/
               │       └── <data_state>/     # e.g., cleaned, raw
               │           ├── csv/
               │           │   ├── history_with_smell.csv
               │           │   └── history_without_smell.csv
               │           │
               │           └── plots/
               │               ├── comparisons/
               │               │   └── <plot_name>.png
               │               ├── moustaches/
               │               │   └── <plot_name>.png
               │               ├── time/
               │               │   └── <plot_name>.png
               │               └── violins/
               │                   └── <plot_name>.png
               │
               └── results/
                   └── <data_state>/         # e.g., cleaned, raw
                       └── <profiler>/       # e.g., mac, carbon
                           ├── <profiler>_report.md
                           ├── with_smell.csv
                           └── without_smell.csv
```

## Column Dictionary

| Column    | Type  | Unit    | Description                                                                                             |
|-----------|-------|---------|---------------------------------------------------------------------------------------------------------|
| `i`       | int   |        | Iteration index (0-based)                                                                               |
| `cpu_mj`  | float | mJ      | CPU package energy for this iteration                                                                   |
| `gpu_mj`  | float | mJ      | Integrated GPU energy                                                                                   |
| `ane_mj`  | float | mJ / g | Apple Neural Engine energy (in mJ) for `mac` profile. CodeCarbon CO₂-eq emissions (in g) for `carbon`. |
| `dram_mj` | float | mJ      | DRAM energy                                                                                             |
| `time_s`  | float | s       | Wall-clock execution time for this iteration                                                            |

## Reproduction

Full instructions and scripts are available in the following repository: https://github.com/fstormacq/energyTracer

```bash
./run_experiment.sh <profile> <warmup_iterations> <measurement_iterations>
```

For more details, please refer to the `README.md` in the reproduction repository.

## Citation

<!-- 
If you use this dataset in your research, please cite it as follows, replacing the appropriate fields (e.g., DOI, author name, etc.) with the actual information related to your dataset:

```bibtex
@dataset{stormacq_2026_zenodo,
  author    = {Florian Stormacq},
  title     = {Energy consumption dataset: code smell vs constant in Python loops},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
``` 
-->

## Contact

For questions about this dataset: your@mail.com