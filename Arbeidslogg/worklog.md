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
