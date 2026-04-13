# 🔍 Health Insurance Fraud Detection

![ROC-AUC](https://img.shields.io/badge/ROC--AUC-97.1%25-brightgreen)
![Recall](https://img.shields.io/badge/Recall-83.2%25-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

## 🚀 Live Demo
👉 [Launch App](https://fraud-detection-mm24vn8vupbdnps6amvr5q.streamlit.app)

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | **97.1%** |
| Recall (Fraud class) | **83.2%** |
| Precision (Fraud class) | **66.1%** |
| Training Claims | 558K+ |
| Providers | 5,410 |

## 📌 Project Summary
Engineered provider-level features from 558K+ Medicare claims to detect 
fraudulent insurance providers using XGBoost — deployed as an interactive 
Streamlit app with SHAP explainability.

## 🧠 Key Insights
- Fraudulent providers file **13x more claims** per provider
- Top fraud signals: TotalReimbursed, IP_MaxLOS, IP_ClaimsPerPatient
- SHAP explainability shows exactly why each provider is flagged

## 🛠 Tech Stack
- XGBoost + SHAP + Streamlit + pandas + scikit-learn
- CMS Medicare Provider Utilization Dataset (558K+ claims)
- 41 engineered provider-level features

## 📈 Model Visuals
![Confusion Matrix](confusion_matrix_tuned.png)
![ROC-AUC Curve](roc_auc_curve.png)
![SHAP Importance](shap_importance.png)
![SHAP Beeswarm](shap_beeswarm.png)

## What failed first — and why it mattered

My first attempt trained directly on claim-level data (558K rows). The model hit 91% accuracy immediately — which felt great until I checked the confusion matrix and realised it was predicting "not fraud" for nearly everything. With only 9.3% fraud rate, always saying "legitimate" gives you 90.7% accuracy for free. The model had learned to do exactly that.

That failure forced the insight that fraud is not a claim-level signal — it is a provider-level behavioural pattern. Fraudulent providers file 13x more claims per provider than legitimate ones. Once I aggregated to provider level and engineered 41 behavioural features, the model learned something real.

## Model comparison

| Model | Recall (fraud) | ROC-AUC | Notes |
|-------|---------------|---------|-------|
| Logistic Regression | 61.4% | 0.891 | Misses non-linear patterns |
| Random Forest | 74.3% | 0.951 | Strong but no sequential correction |
| **XGBoost** | **83.2%** | **0.971** | Best — sequential boosting handles imbalance better |

XGBoost outperformed Random Forest because it builds trees sequentially, each correcting the previous tree's errors — better suited to the highly imbalanced fraud class than parallel ensemble methods.

## Relevance to Indian healthcare

While this project uses the US CMS Medicare dataset (the largest publicly available labelled fraud dataset), the provider-level fraud patterns it detects — upcoding, patient mills, phantom billing — are directly applicable to Indian contexts:

- **PMJAY (Ayushman Bharat)** — India's largest health insurance scheme has reported significant provider-level fraud through inflated claims and ghost beneficiaries
- **IRDAI fraud taxonomy** — matches the same categories: false claims, over-servicing, unnecessary procedures
- The feature engineering approach (aggregating claims by provider, flagging outlier billing behaviour) transfers directly to any claims database regardless of country

Next iteration: retrain on PMJAY-style synthetic data to localise the model for Indian provider fraud patterns.
