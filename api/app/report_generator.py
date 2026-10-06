"""
Clinical report generator.

Builds a human-readable Markdown clinical decision-support report from a
prediction, using the template in ``app/templates/risk_report_template.md``.
The report explains the risk level, the key contributing factors, any model
feature importance, and tailored recommendations — while clearly stating it
is decision-support only, not a diagnosis.
"""
from pathlib import Path

# Map raw STD column names to friendly display names for the report.
_STD_DISPLAY_NAMES = {
    "STDs:HPV": "HPV (Human Papillomavirus)",
    "STDs:HIV": "HIV",
    "STDs:condylomatosis": "Genital condylomatosis",
    "STDs:cervical condylomatosis": "Cervical condylomatosis",
    "STDs:vaginal condylomatosis": "Vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis": "Vulvo-perineal condylomatosis",
    "STDs:syphilis": "Syphilis",
    "STDs:pelvic inflammatory disease": "Pelvic inflammatory disease",
    "STDs:genital herpes": "Genital herpes",
    "STDs:molluscum contagiosum": "Molluscum contagiosum",
    "STDs:AIDS": "AIDS",
    "STDs:Hepatitis B": "Hepatitis B",
}


def _load_template() -> str:
    """Read the Markdown template shipped with the service."""
    template_path = (
        Path(__file__).resolve().parent / "templates" / "risk_report_template.md"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_key_factors(patient_features: dict) -> list[str]:
    """Return a list of Markdown bullets for the risk factors present."""
    lines: list[str] = []

    for key, display in _STD_DISPLAY_NAMES.items():
        if patient_features.get(key):
            lines.append(f"- **{display}:** Positive.")

    if patient_features.get("Dx:HPV"):
        lines.append(
            "- **Prior HPV Diagnosis:** Positive — history of HPV is a significant risk factor."
        )
    if patient_features.get("Dx:CIN"):
        lines.append(
            "- **Prior CIN (Cervical Intraepithelial Neoplasia):** Positive — indicates previous cervical abnormality."
        )
    if patient_features.get("Dx:Cancer"):
        lines.append(
            "- **Prior Cancer Diagnosis:** Positive — history of cancer may indicate higher susceptibility."
        )
    if patient_features.get("Hinselmann"):
        lines.append(
            "- **Hinselmann Test:** Positive — colposcopic finding suggestive of abnormality."
        )
    if patient_features.get("Schiller"):
        lines.append(
            "- **Schiller Test:** Positive — iodine test suggesting abnormal cervical tissue."
        )
    if patient_features.get("Citology"):
        lines.append(
            "- **Cytology (Pap Smear):** Abnormal — previous abnormal screening result noted."
        )

    if not lines:
        lines.append(
            "- No significant risk factors were identified from the available features."
        )
    return lines


def _format_feature_importance(feature_importance) -> str:
    """
    Format the top important features as a Markdown list.

    Expected format: [{"feature": "Age", "importance": 0.25}, ...]
    """
    if not feature_importance:
        return "- Feature importance data is not available for this model."

    lines = []
    for item in feature_importance:
        if isinstance(item, dict):
            feature = item.get("feature") or item.get("name") or "Unknown"
            importance = item.get("importance", item.get("value", 0))
            value = float(importance) * 100
            lines.append(f"- **{feature}:** {value:.1f}%")
    return (
        "\n".join(lines)
        if lines
        else "- Feature importance data is not available for this model."
    )


def _build_recommendations(risk_level: str, patient_features: dict) -> str:
    """Build tailored recommendations based on risk level and patient features."""
    recommendations = []

    if risk_level == "High Risk":
        recommendations.append(
            "- **Consult a Gynecologist:** It is recommended that the patient schedule an appointment with a gynecologist for further evaluation."
        )
        recommendations.append(
            "- **Diagnostic Testing:** Consider Pap smear and/or HPV DNA testing to investigate further. Colposcopy may be indicated."
        )
        recommendations.append(
            "- **Follow-up:** Close monitoring and follow-up as advised by the treating physician is strongly recommended."
        )
    else:
        recommendations.append(
            "- **Routine Screening:** Continue regular cervical cancer screening as per standard guidelines (e.g., Pap smear every 3 years)."
        )

    smokes_years = (
        patient_features.get("Smokes (years)") or patient_features.get("smokes_years") or 0
    )
    if float(smokes_years) > 0:
        recommendations.append(
            "- **Smoking Cessation:** Smoking is a known risk factor for cervical cancer. Smoking cessation support and counseling are recommended."
        )

    recommendations.append(
        "- **Healthy Lifestyle:** Maintain a healthy diet, regular exercise, and practice safe sex (e.g., condom use) to reduce risk."
    )
    recommendations.append(
        "- **HPV Vaccination:** If not already vaccinated, discuss HPV vaccination with the healthcare provider as a preventive measure."
    )

    return "\n".join(recommendations)


def generate_risk_report(
    prediction: int,
    probability: float,
    patient_features: dict,
    feature_importance=None,
    model_name: str = "Unknown",
) -> str:
    """
    Generate a clinical decision-support Markdown report for cervical cancer risk.

    Parameters
    ----------
    prediction : int
        Predicted class (1 = Positive / High Risk, 0 = Negative / Low Risk).
    probability : float
        Confidence / probability of the positive class (0-1).
    patient_features : dict
        The validated patient features (original column names).
    feature_importance : optional
        Optional list of feature-importance dicts.
    model_name : str
        Name of the model that made the prediction.
    """
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    probability_pct = round(float(probability) * 100)

    key_factors_text = "\n".join(_extract_key_factors(patient_features))
    feature_importance_text = _format_feature_importance(feature_importance)
    recommendations_text = _build_recommendations(risk_level, patient_features)

    template = _load_template()
    report = template.format(
        risk_level=risk_level,
        probability=probability_pct,
        model_name=model_name,
        key_factors=key_factors_text,
        feature_importance_list=feature_importance_text,
        recommendations=recommendations_text,
    )
    return report

