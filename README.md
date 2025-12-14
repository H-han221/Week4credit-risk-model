# Week4credit-risk-model
the Week 4 challenge
## Credit Scoring Business Understanding

**Context & objective**  
Bati Bank will use transactional and behavioral data from an eCommerce partner to support a buy-now-pay-later product. Because historical loan repayment outcomes are not available, we create a defensible proxy for “high-risk” borrowers using RFM (Recency, Frequency, Monetary) clustering and use that proxy to build supervised models that estimate a customer’s probability of being high risk.

**How the Basel II Accord’s emphasis on risk measurement influences model requirements**  
Basel II requires banks to measure, quantify and hold capital against credit risk, and it places strong emphasis on model governance, validation, and robust documentation. In practice this means our credit model must be reproducible, versioned, and auditable — with clear documentation of data sources, feature construction, model development, and validation results — so risk officers and regulators can assess model assumptions, stability, and capital implications. Prioritize explainability and traceability as part of the minimum compliance and governance controls. :contentReference[oaicite:0]{index=0}

**Why a proxy target is necessary and the business risks of relying on it**  
A supervised credit model requires examples of “good” and “bad” outcomes. Because the dataset lacks historical loan repayment labels, we must define a proxy (for example, a cluster of disengaged customers determined by RFM metrics) to label high-risk cases for training. This enables model development but creates risks:  
- **Proxy mismatch:** the proxy may not perfectly correlate with true default risk; the model may learn patterns that predict the proxy but not actual defaults.  
- **Selection and fairness bias:** proxy construction may inadvertently encode socio-demographic or product behavior that is not causal for creditworthiness, leading to unfair treatment of subpopulations.  
- **Stability and drift:** customer behavior that defines the proxy can change (seasonality, promotions, macro shocks), requiring monitoring and recalibration.  
To mitigate these risks, document proxy logic, run sensitivity and bias analyses, combine model output with business rules and manual review for borderline cases, and implement ongoing performance/drift monitoring. :contentReference[oaicite:1]{index=1}

**Trade-offs: simple interpretable model (Logistic + WoE) vs. complex high-performance model (Gradient Boosting)**  
- **Interpretable models (Logistic Regression + WoE/scorecards)**  
  - *Pros:* transparent, easier to explain to regulators, stable after WoE binning, straightforward to convert probabilities to a scorecard, simpler validation and documentation.  
  - *Cons:* limited ability to capture complex non-linear interactions; may underperform versus more flexible learners on heterogeneous behavioral signals.  
- **Complex models (Gradient Boosting, XGBoost, LightGBM)**  
  - *Pros:* typically higher predictive power, can model nonlinearity and feature interactions automatically, often improves discrimination on alternative data.  
  - *Cons:* harder to interpret (requires SHAP/LIME or surrogate explainers), increased operational risk from overfitting, and a higher burden on governance and monitoring to satisfy regulatory scrutiny.  
**Recommended pragmatic approach:** produce an interpretable, well-documented scorecard for primary regulatory and decisioning use, while developing an advanced model in parallel as a benchmark and adjudication tool for edge cases. Ensure both are tracked, versioned, and subject to the same validation and monitoring pipelines. :contentReference[oaicite:2]{index=2}

### References
- Huang, D., Zhou, J., & Wang, H., *RFMS Method for Credit Scoring Based on Bank Card Transaction Data* (Statistica Sinica). :contentReference[oaicite:3]{index=3}  
- World Bank, *Credit Scoring Approaches: Guidelines* (2020). :contentReference[oaicite:4]{index=4}  
- Corporate Finance Institute, *Credit Risk* (reference article). :contentReference[oaicite:5]{index=5}  
- Towards Data Science, *How to Develop a Credit Risk Model and Scorecard*. :contentReference[oaicite:6]{index=6}

