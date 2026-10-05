# Data provenance

Both CSV files were copied unchanged from the completed course project. Duplicate copies of the Telco data were consolidated into one file.

## Telco customer churn

- File: `telco_churn.csv`, renamed from `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- Rows: 7,043; columns: 21, including customerID and Churn.
- Source attribution: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- This release uses the course-provided copy, not a newly downloaded dataset. The Kaggle page is the cited background source; an independent upstream byte-for-byte comparison was not performed.
- SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.

## Bitcoin prices

- File: `btc_price.csv`.
- Source: supplied by the professor for the course project, as confirmed by the contributors.
- Original upstream website and distribution terms were not recorded in the project. No guessed provider is attributed.
- Rows: 1650; date coverage: 2019-03-08–2023-11-22.
- Columns: Date, Close/Last, Volume, Open, High, Low. Modeling uses Date and High only.
- Seven intervals span more than one calendar day; 30-observation windows are not necessarily 30 calendar days. Invalid values are handled by the scripts; the CSV itself was not rewritten.
- SHA-256: `d520173d626504d73a1fc08e0dc835aaf4b88f7892dbeba213c86a00ae4bc712`.

These provided datasets have not been assigned a new license by this release. The repository records their known provenance without claiming independently verified redistribution rights.
