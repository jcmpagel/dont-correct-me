# Don't Correct Me: Faithful Transcription of Contemporary Handwriting

Jonathan Pagel, Khaoula Otmani, and Enkelejda Kasneci

Data, model transcriptions, and evaluation code for [the paper](https://openreview.net/forum?id=bFQGWlMPoz).

| Dataset | Images | Model configurations | Transcriptions |
|---|---:|---:|---:|
| `benchmark/` | 845 | 28 | 23,660 |
| `nonwords/` | 1,000 | 28 | 28,000 |

The benchmark contains English and German handwriting with 953 annotated misspellings, of which 947 have a uniquely resolved target position. The nonword test contains lines assembled from 5,500 GoBo word images contributed by 40 writers.

## Files

Each dataset directory contains `images/`, `annotations.csv`, and `predictions.csv`. Image paths are relative to the dataset directory. The `sample_id` is the image filename without its extension (for example, `prolific_de_0001` for `images/prolific_de_0001.png`). Join annotations and predictions using `sample_id`.

Benchmark annotations contain the literal reference (`original_transcript`), corrected transcript, misspellings (`flagged_words`), accepted corrections (`canonical_corrections`, `accepted_alternatives`), source, language, and bootstrap cluster. Multiple targets and corrections are separated by `|` in corresponding order. Six repeated-form targets have `correction_status=unresolved_target`; their images remain in the CER/WER evaluation.

Nonword annotations contain the literal reference, source, writer identifier, and word count.

Both prediction files use the same columns:

| Column | Meaning |
|---|---|
| `sample_id` | Annotation identifier |
| `model_name` | Model configuration used in the paper |
| `provider` | Provider route |
| `model` | Model identifier |
| `prediction_text` | Model transcription |
| `successful` | 1 for successful output, 0 otherwise |
| `error` | Recorded error, if present |

CSV files use UTF-8 and can be read with Python's `csv` module or `pandas.read_csv`. Empty and failed outputs are included.

## Evaluation

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python evaluation/benchmark.py
python evaluation/nonwords.py
```

The benchmark script calculates CER, WER, exact match, misspelling outcomes, source/language breakdowns, scoring-policy sensitivity, and paired confidence intervals (10,000 bootstrap resamples; seed 27). The nonword script calculates nonword preservation, SCOWL lexicality, and correlations with the benchmark results. Run the benchmark script first. Outputs are written to `results/benchmark/` and `results/nonwords/`.

Evaluation uses the supplied transcriptions and runs offline. Target outcomes use minimum-edit character alignment with word-boundary expansion and case-insensitive comparison of the extracted target spans. Nonword preservation uses normalised token matching; lexicality uses the supplied British/American SCOWL word list.

## Sources and terms

Source licenses apply to the corresponding data. New PEER and Prolific crops and benchmark annotations are provided under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Upstream rights are retained.

| Source | Use | Citation and terms |
|---|---|---|
| SMHD | 297 benchmark images | Nisa, Thom, Ciesielski, and Tennakoon (2023), [Student Handwriting Misspelling Dataset](https://doi.org/10.25439/RMT.24312715.V1); CC BY-NC 4.0 |
| GNHK | 14 benchmark images | Lee, Chung, and Lee (2021), [GNHK: A Dataset for English Handwriting in the Wild](https://github.com/GoodNotes/GNHK-dataset); CC BY 4.0 |
| Handwritten ASAP-SAS | 31 benchmark images | Gold and Zesch (2020), [Exploring the Impact of Handwriting Recognition on the Automated Scoring of Handwritten Student Answers](https://doi.org/10.5281/zenodo.8088866); CC BY 4.0 |
| FH-SWF German Handwriting | 23 benchmark images; German stimuli | Fachhochschule Südwestfalen, [German Handwriting](https://huggingface.co/datasets/fhswf/german_handwriting); AFL-3.0 |
| PEER | 32 benchmark images | Bühler, Bueno, and Kasneci (2026), [Democratizing Writing Support with AI](https://doi.org/10.1609/aaai.v40i45.41167); released crops: CC BY-NC 4.0 |
| Prolific German and English studies | 349 German and 99 English images | Pagel, Otmani, and Kasneci, [Don't Correct Me](https://openreview.net/forum?id=bFQGWlMPoz); CC BY-NC 4.0 |
| GoBo v1.0 | Word images for the nonword test | Gold, van den Boom, and Zesch (2021), [Personalizing Handwriting Recognition Systems with Limited User-Specific Samples](https://doi.org/10.1007/978-3-030-86337-1_28); [dataset and terms](https://doi.org/10.5281/zenodo.8085511), non-commercial research use |
| MERLIN | German stimuli | Boyd et al. (2014), [MERLIN](https://www.merlin-platform.eu/C_data.php); CC BY-SA 4.0 |
| Litkey | German stimuli | Laarmann-Quante et al. (2019), [The Litkey Corpus](https://doi.org/10.3758/s13428-019-01261-x); source-specific terms |
| REALEC | English stimuli | Kuzmenko and Kutuzov (2014), [Russian Error-Annotated Learner English Corpus](https://realec.org/about/); adapted stimulus text: CC BY-SA 4.0 |
| SCOWL / ESDB | English lexicality | Kevin Atkinson and contributors, [revision 1e5b7d3](https://github.com/en-wl/wordlist/tree/1e5b7d3a72f47a71da5d28686c1dd4b397178485); component notices in [COPYRIGHT.txt](evaluation/lexicon/COPYRIGHT.txt) |
