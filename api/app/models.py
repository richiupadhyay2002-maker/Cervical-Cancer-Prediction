"""
Pydantic schemas for request validation and response serialisation.

All 35 clinical features from the cervical-cancer dataset are defined with a
type, a valid range, and an alias that maps the JSON key back to the original
column name used during model training. Pydantic v2 enforces these rules so
invalid or malicious payloads are rejected before they reach the model.
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictionInput(BaseModel):
    """
    Input features for a cervical cancer risk prediction.

    Each field maps to one of the 35 features used during model training.
    Fields are Optional so callers only need to send the values they have;
    ``extra="forbid"`` rejects any unknown/misspelled field.
    """

    age: int = Field(
        ...,
        alias="Age",
        ge=10,
        le=100,
        description="Patient's age in years",
    )
    number_of_sexual_partners: Optional[int] = Field(
        None, alias="Number of sexual partners", ge=0, le=100
    )
    first_sexual_intercourse: Optional[int] = Field(
        None, alias="First sexual intercourse", ge=10, le=80
    )
    num_of_pregnancies: Optional[int] = Field(
        None, alias="Num of pregnancies", ge=0, le=20
    )
    smokes: Optional[int] = Field(None, alias="Smokes", ge=0, le=1)
    smokes_years: Optional[int] = Field(None, alias="Smokes (years)", ge=0, le=50)
    smokes_packs_per_year: Optional[float] = Field(
        None, alias="Smokes (packs/year)", ge=0.0, le=100.0
    )
    hormonal_contraceptives: Optional[int] = Field(
        None, alias="Hormonal Contraceptives", ge=0, le=1
    )
    hormonal_contraceptives_years: Optional[float] = Field(
        None, alias="Hormonal Contraceptives (years)", ge=0.0, le=30.0
    )
    iud: Optional[int] = Field(None, alias="IUD", ge=0, le=1)
    iud_years: Optional[float] = Field(None, alias="IUD (years)", ge=0.0, le=30.0)
    stds: Optional[int] = Field(None, alias="STDs", ge=0, le=1)
    stds_number: Optional[int] = Field(
        None, alias="STDs (number)", ge=0, le=10
    )
    # --- 12 individual STD flags (0/1) -------------------------------------
    stds_condylomatosis: Optional[int] = Field(
        None, alias="STDs:condylomatosis", ge=0, le=1
    )
    stds_cervical_condylomatosis: Optional[int] = Field(
        None, alias="STDs:cervical condylomatosis", ge=0, le=1
    )
    stds_vaginal_condylomatosis: Optional[int] = Field(
        None, alias="STDs:vaginal condylomatosis", ge=0, le=1
    )
    stds_vulvo_perineal_condylomatosis: Optional[int] = Field(
        None, alias="STDs:vulvo-perineal condylomatosis", ge=0, le=1
    )
    stds_syphilis: Optional[int] = Field(None, alias="STDs:syphilis", ge=0, le=1)
    stds_pelvic_inflammatory_disease: Optional[int] = Field(
        None, alias="STDs:pelvic inflammatory disease", ge=0, le=1
    )
    stds_genital_herpes: Optional[int] = Field(
        None, alias="STDs:genital herpes", ge=0, le=1
    )
    stds_molluscum_contagiosum: Optional[int] = Field(
        None, alias="STDs:molluscum contagiosum", ge=0, le=1
    )
    stds_aids: Optional[int] = Field(None, alias="STDs:AIDS", ge=0, le=1)
    stds_hiv: Optional[int] = Field(None, alias="STDs:HIV", ge=0, le=1)
    stds_hepatitis_b: Optional[int] = Field(
        None, alias="STDs:Hepatitis B", ge=0, le=1
    )
    stds_hpv: Optional[int] = Field(None, alias="STDs:HPV", ge=0, le=1)
    stds_number_of_diagnosis: Optional[int] = Field(
        None, alias="STDs: Number of diagnosis", ge=0, le=10
    )
    stds_time_since_first_diagnosis: Optional[float] = Field(
        None, alias="STDs: Time since first diagnosis", ge=0.0, le=50.0
    )
    stds_time_since_last_diagnosis: Optional[float] = Field(
        None, alias="STDs: Time since last diagnosis", ge=0.0, le=50.0
    )
    # --- 4 prior-diagnosis flags (0/1) --------------------------------------
    dx_cancer: Optional[int] = Field(None, alias="Dx:Cancer", ge=0, le=1)
    dx_cin: Optional[int] = Field(None, alias="Dx:CIN", ge=0, le=1)
    dx_hpv: Optional[int] = Field(None, alias="Dx:HPV", ge=0, le=1)
    dx: Optional[int] = Field(None, alias="Dx", ge=0, le=1)
    # --- 3 screening-test outcomes (0/1) -------------------------------------
    hinselmann: Optional[int] = Field(None, alias="Hinselmann", ge=0, le=1)
    schiller: Optional[int] = Field(None, alias="Schiller", ge=0, le=1)
    citology: Optional[int] = Field(None, alias="Citology", ge=0, le=1)

    @field_validator(
        "smokes",
        "hormonal_contraceptives",
        "iud",
        "stds",
        "stds_condylomatosis",
        "stds_cervical_condylomatosis",
        "stds_vaginal_condylomatosis",
        "stds_vulvo_perineal_condylomatosis",
        "stds_syphilis",
        "stds_pelvic_inflammatory_disease",
        "stds_genital_herpes",
        "stds_molluscum_contagiosum",
        "stds_aids",
        "stds_hiv",
        "stds_hepatitis_b",
        "stds_hpv",
        "dx_cancer",
        "dx_cin",
        "dx_hpv",
        "dx",
        "hinselmann",
        "schiller",
        "citology",
    )
    @classmethod
    def validate_binary(cls, v):
        """Ensure binary flags are exactly 0 or 1."""
        if v is not None and v not in (0, 1):
            raise ValueError("Must be 0 or 1")
        return v

    class Config:
        populate_by_name = True
        extra = "forbid"


class PredictionOutput(BaseModel):
    """Response returned by the /predict endpoint."""

    model_name: str = Field(..., description="Name of the registered model")
    model_version: int = Field(..., description="Version of the model used")
    prediction: int = Field(..., description="Predicted class (0 = Negative, 1 = Positive)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Prediction probability (if available)"
    )
    threshold: float = Field(0.5, description="Decision threshold used")


class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""

    status: str = Field("ok", description="Service health status")
    model_name: str = Field(..., description="Name of the loaded model")
    model_version: int = Field(..., description="Version of the loaded model")
    model_stage: str = Field(..., description="Stage the model was loaded from")
    features_count: int = Field(..., description="Number of expected input features")


class ReportOutput(BaseModel):
    """Response returned by the /predict/report endpoint."""

    model_name: str = Field(..., description="Name of the registered model")
    model_version: int = Field(..., description="Version of the model used")
    prediction: int = Field(..., description="Predicted class (0 = Negative, 1 = Positive)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Prediction probability (if available)"
    )
    risk_level: str = Field(..., description="High Risk or Low Risk")
    report_markdown: str = Field(..., description="Generated Markdown clinical report")


class ErrorResponse(BaseModel):
    """Standard error response for 4xx/5xx errors."""

    error: str = Field(..., description="Error type or title")
    detail: Optional[str] = Field(None, description="Detailed error message")

