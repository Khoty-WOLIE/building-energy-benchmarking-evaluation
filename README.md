# Within-Building Dependence, Target Leakage and Data Vintages in Building Energy Prediction

Code, results and LaTeX source of the preprint

> Khoty Wolie (2026). *Within-Building Dependence, Target Leakage and Data Vintages in Building Energy Prediction
> from Public Benchmarking Data. Evidence from Seattle, 2015 to 2025.* arXiv preprint (identifier added once announced).

The study analyses Seattle's non-residential building energy benchmarking panel (15,621 building-years, 1,667
buildings, 2015 to 2025) and two releases of the 2016 data. It shows that row-random evaluation strongly overstates
accuracy on unseen buildings, that the ENERGY STAR score leaks the target, and that data revisions change absolute
error levels across data vintages.

## Repository layout

| Path | Content |
|---|---|
| `experiments/02_protocole_experimental_v1.0.md` | Pre-registered protocol, frozen on 2026-10-29, with its deviation log (written in French) |
| `experiments/config/protocol_v1.yaml` | Machine-readable version of the protocol |
| `src/a01/` | Data preparation, information tiers, evaluation splits, models, statistics and run provenance |
| `src/run_panel.py`, `src/run_e6_vintage.py`, `src/run_exploratory.py` | Experiments E1 to E6 and exploratory analyses X1 and X2 |
| `src/analyze_results.py`, `src/make_tables.py`, `src/make_figures.py` | Primary contrasts with Holm correction, analysis X3, tables and figures |
| `src/check_manuscript.py`, `src/build_manuscript_bundle.py` | Recompute every number quoted in the paper from the results, then compile the PDF |
| `src/profile_dataset.py`, `src/compare_2016_versions.py`, `src/replicate_original_pipeline.py`, `src/prepare_phase03.py` | Data audit, comparison of the two 2016 releases, reproduction of the original exercise, population and split summaries |
| `tests/` | Unit tests of split invariants and statistics |
| `results/` | Result tables and per-run provenance records (configuration, data hashes, environment) |
| `manuscript/` | LaTeX source of the paper, generated tables and figures |
| `literature/` | Verified bibliography and the script that checks every reference against Crossref and arXiv |

Per-run `commit.txt` files refer to the author's private development history. Per-row prediction files are not
stored here because the code regenerates them.

## Data

Raw data are not redistributed. Download them into `data/raw/` and check their SHA-256 against
`data/raw/SHA256SUMS.txt`.

- **Consolidated panel** (public domain). City of Seattle, *Building Energy Benchmarking Data, 2015-Present*,
  dataset `teqw-tu6e`, from `https://data.seattle.gov/api/views/teqw-tu6e/rows.csv?accessType=DOWNLOAD`, saved as
  `data/raw/seattle_benchmarking_2015_present_teqw-tu6e.csv`. The portal is updated continuously, and the code
  refuses any file other than the snapshot of 2026-10-29.
- **Original 2016 release** (former dataset `2bpz-gwpy`, no longer served by the city portal), saved as
  `data/raw/2016_Building_Energy_Benchmarking.csv`. Its provenance is described in the paper.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt        # on Windows use .venv\Scripts\pip
python -m pytest tests -q
python src/run_e6_vintage.py data/raw/seattle_benchmarking_2015_present_teqw-tu6e.csv data/raw/2016_Building_Energy_Benchmarking.csv
python src/run_panel.py E1
python src/run_panel.py E2
python src/run_panel.py E3
python src/run_panel.py E5
python src/run_exploratory.py X1
python src/run_exploratory.py X2
python src/analyze_results.py
python src/make_tables.py
python src/make_figures.py
python src/check_manuscript.py
python src/build_manuscript_bundle.py --pdf          # requires a LaTeX distribution
```

All experiments take about 75 minutes on a 24-core CPU.

## Acknowledgements

The author used generative AI tools as research assistants, for literature searches, programming support and text
editing. The author designed the study, made all scientific decisions, verified the results and takes full
responsibility for the content.

## License

Code is released under the MIT License (see `LICENSE`). The paper is distributed on arXiv under CC BY 4.0.

## Citation

See `CITATION.cff`. Contact: khotywolie@gmail.com
