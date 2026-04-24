# Work Log - LOG650 AI Forecasting Project

---

## 24.02.2026
- Cloned repository locally.
- Inserted approved proposal.
- Added Word report draft.
- Set up GitHub workflow.
- Completed full introduction section in report.
- Added background, research question, and scope clarification.

## 25.02.2026
- Simulated dataset for PSV vessels (historical on-hire / off-hire periods).
- Defined assumptions behind simulation (technical failure vs. market unavailability).
- Structured dataset variables for further quantitative analysis.
- Drafted literature review section in report.
- Identified recent contributions related to AI-based forecasting in logistics.

## Mars 2026
- Expanded the report substantially beyond the February draft.
- Developed the report structure with chapters for introduction, literature, theory, case description, method, data, modeling, analysis, results, discussion, conclusion, bibliography, and appendix.
- Wrote out the problem framing around forecasting offhire events in the PSV segment and clarified delimitations and assumptions for the study.
- Added a broader literature review on demand forecasting, time-series methods, machine learning, robustness, interpretability, and recent forecasting competitions.
- Added a theory section that positions the thesis in the intersection between forecasting as decision support, model choice under uncertainty, and AI use in maritime and offshore operations.
- Added a case description centered on Simon Møkster Shipping / PSV operations, offhire as an operational and commercial availability problem, and the relevance of comparing traditional models with AI-based models.
- Continued drafting the methodological and analytical parts of the report, while leaving parts of the method/data/modeling sections as placeholders for later completion.

## 13.04.2026
- Converted the active report draft to Markdown to make the report easier to version and edit alongside the repository.
- Standardized in-text citations in the Markdown report toward APA 7 style.
- Defined the final model set to be compared in the thesis: SARIMA, Exponential Smoothing, XGBoost, and LSTM.
- Built an initial modeling pipeline under `004 data/modeling` with preprocessing, feature engineering, train/test split, model runners, and result export.
- Added environment setup and usage documentation for the modeling pipeline.
- Tested the models on the anonymized dataset currently available in the repository.
- Confirmed that Exponential Smoothing, XGBoost, and LSTM run on the current dataset, while SARIMA is not methodologically defensible yet because the time series is too short for a seasonal monthly model.
- Cleaned up the Git worktree, moved only the necessary modeling files into version control, and pushed the changes to GitHub.
- Structured the modeling folder so each model now has separate code and result logs for systematic documentation.
- Added automatic generation of model-specific result files and a combined comparison file to support the final report writing.
- Included the anonymized dataset, generated modeling outputs, and documentation files in the repository so the full workflow is traceable for delivery.

Next step:
- Continue writing the method, data, and modeling chapters so they reflect the implemented pipeline and the actual dataset limitations.
- Assess whether more historical observations are needed before SARIMA can be included in the final empirical comparison.
- Keep updating the report text and bibliography as additional references are added.

## 14.04.2026
- Merged and synchronized `main` with GitHub, including the new `005 report` folder and the latest repository state.
- Replaced the old simulated dataset setup with the updated anonymized operational dataset covering April 2021 to March 2026.
- Updated the data parser in the modeling pipeline so the code correctly reads the multi-year CSV structure with annual blocks from 2021 to 2026.
- Restricted the active model run to Exponential Smoothing while keeping the other model implementations in the codebase for later comparison.
- Reorganized the modeling output so model-specific files are grouped under dedicated folders instead of being spread across the modeling directory.
- Built a separate historical visualization pipeline under `004 data/visualization` and generated figures and summary tables for the descriptive data analysis.
- Inserted the key historical figures and explanatory text directly into the report, and refined the report text so the case is described as 16 vessels operating within the same offshore segment.
- Identified that the previous evaluation setup was too weak, and rebuilt the modeling workflow around an explicit time-based train/test split.
- Added `004 data/processed/train.csv` and `004 data/processed/test.csv` in the same raw format as the master dataset, with train covering `2021-04` to `2024-12` and test covering `2025-01` to `2026-03`.
- Updated the modeling code so historical evaluation now runs on the explicit train/test split, while future forecasts still use the full history up to March 2026.
- Regenerated Exponential Smoothing results under the new split and documented the revised evaluation setup in both the report and modeling README.
- Committed and pushed the main implementation changes to GitHub in commits `d336d02` (`Update analysis assets and add historical visualizations`) and `8dfcf9d` (`Add explicit train-test split for model evaluation`).
- Reworked the modeling chapter around a shared six-step structure so each model is now described through model-specific stages from data preparation to validation and forecasting.
- Expanded the report with separate subsections for `SARIMA`, `Eksponentiell glatting`, `XGBoost`, and `LSTM`, while keeping the detailed diagnostics and technical artifacts in the repository.
- Re-enabled all four models in the pipeline so the standard run now evaluates the full model set on the same train/test split.
- Implemented a Box-Jenkins-style `SARIMA` workflow on fleet level with stationarity testing, ACF/PACF diagnostics, candidate model selection, residual diagnostics, and documented model choice.
- Added automatic generation of `metode.md` for each model and extra SARIMA artifacts such as stationarity tables, candidate model tables, and diagnostic plots.
- Ran the full pipeline on the current dataset and confirmed that all four models complete under the shared evaluation setup, with updated comparison outputs written to the modeling folders.

## 15.04.2026
- Reviewed the lecturers' example structure and used it to tighten the report logic around literature, theory, modeling, and results.
- Reworked the report so `Litteratur` now functions as a research review, while `Teori` is structured around the four actual model families used in the thesis: `SARIMA`, `Eksponentiell glatting`, `XGBoost`, and `LSTM`.
- Updated the theory chapter to rely on more established forecasting and model references, and aligned the writing more clearly with APA 7 style in the report.
- Rebuilt the modeling workflow so all four models are evaluated on the same historical setup at vessel level with a shared train/test split and expanding `1-step` evaluation through the full test period.
- Refactored `SARIMA` from a fleet-level setup to a per-vessel `ARIMA/SARIMA` workflow with stationarity support, bounded candidate search, model selection tables, residual diagnostics, and representative validation plots.
- Refactored `Eksponentiell glatting` into a comparable per-vessel benchmark with explicit ETS specification selection, residual diagnostics, and representative test plots.
- Refactored `XGBoost` into a cleaner panel-data workflow with lag features, rolling features, calendar features, vessel encoding, feature importance outputs, and month-by-month walk-forward evaluation.
- Refactored `LSTM` into a shared sequence-based evaluation workflow with `12`-month input windows, train-only scaling, training-history artifacts, and representative test plots.
- Added combined verification outputs under `004 data/modeling/outputs/shared`, including summary tables by model, vessel, and month, plus figure outputs for `MAE` comparison and vessel-level error heatmaps.
- Removed `future_predictions.csv` from the standard modeling run so the default pipeline now documents only model building, testing, and verification, while future forecasting is left for a later report section.
- Updated the modeling README and the report chapters so they match the revised workflow and explicitly use figures/tables as verification material for each model.
- Ran the full pipeline after the refactor and confirmed the latest shared historical test ranking: `SARIMA` best, followed by `XGBoost`, `LSTM`, and `Eksponentiell glatting`.
- Committed and pushed the main work in `d3186e6` (`rewrite theory and literature sections`) and `5ad242a` (`refine model testing and verification workflow`).

## 16.04.2026
- Extended the modeling pipeline so future forecasts are now generated systematically for `1`, `3`, `6`, and `12` months ahead from the latest observed month in the dataset.
- Reintroduced future forecast outputs under `004 data/modeling/outputs/shared` with combined forecast tables, horizon-specific tables, pivot tables, and updated forecast figures for use in the report.
- Reworked the report structure from `Casebeskrivelse` through `Resultat` so the historical material is split more clearly between case context, data description, modeling, and results.
- Moved the descriptive historical analysis into `Metode og data`, added clearer subsections for `Datagrunnlag` and `Deskriptiv analyse av datasettet`, and removed the previous standalone analysis chapter.
- Revised `Litteratur` and `Teori` so the text is more technically balanced across `SARIMA`, `Eksponentiell glatting`, `XGBoost`, and `LSTM`, while keeping the report aligned with APA 7.
- Moved the thesis-specific model selection rationale and comparison setup out of theory and into the methods section so the distinction between theory and method is clearer.
- Wrote the discussion chapter as a full analytical section with subsections on model choice, data structure, future forecasts, methodological strengths and weaknesses, and practical implications for Simon Møkster Shipping AS.
- Strengthened the case description and discussion with explicit context about offshore and oil-and-gas market volatility, while clarifying that external market drivers are not modeled directly in the empirical analysis.
- Completed the conclusion section so it now answers the research question directly, summarizes the main empirical findings, and ends with a concise note on further research.
- Added appendix sections with figure overview, table overview, and model-specific code appendices aligned with the four-model comparison in the main report.
- Refactored `004 data/modeling/run_models.py` into a modular pipeline with shared helper modules and one Python file per model: `sarima.py`, `exponential_smoothing.py`, `xgboost_model.py`, and `lstm_model.py`.
- Updated the automatic `kode.md` generation so each model folder now documents the active implementation from its own module instead of from the previous monolithic pipeline file.
- Verified the refactored pipeline with both `py_compile` and a full end-to-end run; the latest historical ranking remained `SARIMA`, `XGBoost`, `LSTM`, and `Eksponentiell glatting`.
- Committed and pushed the main implementation changes in `7ad8661` (`Finalize report and modularize modeling pipeline`).

## 17.04.2026
- Continued the report finishing work with a full structural and editorial review of `Rapport.md`, focusing on overlap between theory, modeling, results, discussion, and conclusion.
- Moved the standard mathematical model formulations out of `Modellering` and into the relevant theory subsections for `SARIMA`, `Eksponentiell glatting`, `XGBoost`, and `LSTM`, so the distinction between theory and project-specific modeling is clearer.
- Refined the report front page and manual table of contents so the Markdown version aligns better with the required project template and the lecturers' example structure.
- Removed the obsolete Word draft `Rapport prøving.docx` from the repository and kept `Rapport.md` as the active report source.
- Opened version control for the full `003 references` folder and pushed all reference PDFs to GitHub to satisfy the delivery requirement that the PDF sources must be available in the repository.
- Tightened the report's theory presentation by removing an unreadable mathematical summary table and keeping the detailed model explanations in running text instead.
- Committed and pushed the main report and repository changes in `23642e6` (`Refine report structure and model theory sections`), `b6e68f8` (`Refine report front page and contents`), `7730b45` (`Remove obsolete report draft document`), `bfafd26` (`Track reference PDFs in repository`), and `6df4621` (`Remove unreadable model summary table`).
