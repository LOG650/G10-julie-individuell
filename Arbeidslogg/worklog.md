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
- Added `004 data/train.csv` and `004 data/test.csv` in the same raw format as the master dataset, with train covering `2021-04` to `2024-12` and test covering `2025-01` to `2026-03`.
- Updated the modeling code so historical evaluation now runs on the explicit train/test split, while future forecasts still use the full history up to March 2026.
- Regenerated Exponential Smoothing results under the new split and documented the revised evaluation setup in both the report and modeling README.
- Committed and pushed the main implementation changes to GitHub in commits `d336d02` (`Update analysis assets and add historical visualizations`) and `8dfcf9d` (`Add explicit train-test split for model evaluation`).
