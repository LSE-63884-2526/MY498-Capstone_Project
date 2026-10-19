This repository contains the scripts, notebooks, and visuals for the capstone project titled “Opposition Discourse under a Competitive Authoritarian Regime: Computational Discourse Analysis of the Republican People’s Party (2015-2026)”. The thesis was submitted in partial fulfilment of the requirements for the degree of MSc in Applied Social Data Science, Department of Methodology, London School of Economics and Political Science.

The embedding vectors and metadata required to reproduce the analysis are available here: [huggingface.co/datasets/athena753/CHP2015_2026](https://huggingface.co/datasets/athena753/CHP2015_2026)

```
capstone/
├── notebooks/
│   ├── demokrasi_analysis.ipynb
│   ├── hak_analysis.ipynb
│   ├── halk_analysis.ipynb
│   ├── lexical_analysis.ipynb
│   └── millet_analysis.ipynb
├── scripts/
│   ├── data_collection.py
│   ├── preprocessing_pipeline.py
│   ├── default_stems.py
│   ├── lexical_analysis.py
│   ├── bert_embeddings_mw.py
│   ├── signifier_analysis.py
│   └── robustness.py
├── visuals/
│   ├── demokrasi_phase_heatmap.pdf
│   ├── hak_phase_heatmap.pdf
│   ├── halk_phase_heatmap.pdf
│   └── millet_phase_heatmap.pdf
├── stopwords_tr.txt
├── supplementary.pdf      # supplementary document
└── LSE498_63884.pdf       # dissertation
```

## Notebooks

The notebooks contain the main analysis. Each one imports the `signifier_analysis` module and follows the same structure:

1. Import the `signifier_analysis` module
2. Run a sensitivity analysis to choose the number of UMAP components
3. Fit the clusters
4. Close reading with the keyword in context (KWIC) function
5. Assign clusters to meta-clusters
   - Split a cluster if necessary (only if the cluster is dominated by the same morphological structure (e.g. hak etmek))
6. Plot the heatmap
7. Check the number of clusters and occurrences assigned to each meta-cluster

## Scripts

| Script | Purpose |
|---|---|
| data_collection.py | Scrapes speeches from chp.org.tr and saves a .csv file with id, venue, speaker, and date. |
| preprocessing_pipeline.py | Cleans the corpus and saves two versions: `text_bert` to produce embeddings and `text_lemma` for dictionary analysis. |
| default_stems.py | Manually defined stemming rules for Turkish words. |
| lexical_analysis.py | Dislocation analysis tracking how economic demand vocabulary enters and recedes. |
| bert_embeddings_mw.py | Produces contextual embeddings for signifiers with masking and whitening. |
| signifier_analysis.py | Main analysis module, imported by each `[signifier]_analysis.ipynb` notebook. Includes functions for:<br>• fitting the clusters<br>• producing heatmaps<br>• close reading via KWIC (keyword in context) |
| robustness.py | Runs silhouette, mask+whiten validation, and stylometry checks. |
