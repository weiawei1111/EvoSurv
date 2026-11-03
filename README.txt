EvoSurv
=======

EvoSurv is an evolution-aware multimodal survival model. It combines RNA expression,
eTime (38 tumor-evolution features in eight groups) and a fixed clinical risk offset
(stage and age).

Files
  features/run_pipeline.py   builds the eTime features from PyClone, MAF and ABSOLUTE data
  model/etime.py             eTime feature groups and preprocessing
  model/rna.py               RNA preprocessing, autoencoder, Cox loss, C-index, data splits
  model/model.py             EvoSurv network and training
  model/run_evosurv.py       data loading and evaluation on repeated splits (main script)

Data
  From the GDC, download somatic mutation MAF files
  (with VEP consequence and SIFT annotations), ABSOLUTE gene-level copy number, purity and
  allele-specific copy number, and RNA-seq expression . Survival, stage and age
  come from TCGA-CDR. Run PyClone on the non-silent coding SNVs of
  each tumor, using ABSOLUTE purity and allele-specific copy number as input. The anoikis
  gene list is the GeneCards result for the query "anoikis" (2026).

  Expected layout under $EVOSURV_DATA:
    rna/<CANCER>_processed_rna.csv
    cohort/<CANCER>_cohort_model_cdr.txt
    cohort/survival/<CANCER>_survival.csv          patient_id, OS_time_days, OS_event
    cohort/raw/TCGA-CDR-SupplementalTableS1.xlsx
    features_v2/<CANCER>/all_features.csv          output of run_pipeline.py

Usage
  pip install numpy pandas scipy torch openpyxl       # Python 3.9
  export EVOSURV_DATA=/path/to/data
  python features/run_pipeline.py --pyclone_v2
  cd model && python run_evosurv.py --out_root ../results

  Each cancer is evaluated on 50 event-stratified 64/16/20 splits (seeds 0-49). Test
  C-indices are written to results/<CANCER>/runs.csv and summarized in results/summary.txt.
   
