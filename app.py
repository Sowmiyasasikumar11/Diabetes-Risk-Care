from functools import wraps
from pathlib import Path
import csv
from io import StringIO
import sqlite3
import math

import joblib
import pandas as pd
from flask import Flask, Response, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import RequestEntityTooLarge

from report_extraction import (
    MAX_REPORT_BYTES,
    ReportProcessingError,
    extract_medical_values,
    extract_report_text,
    is_allowed_filename,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "diabetes.db"
MODEL_PATH = PROJECT_ROOT / "outputs" / "random_forest_model.pkl"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "selected_features.csv"

app = Flask(__name__)
app.config["SECRET_KEY"] = "development-only-change-this-secret"
app.config["DATABASE"] = DATABASE_PATH
app.config["MAX_CONTENT_LENGTH"] = MAX_REPORT_BYTES

# Demo administrator account. Change these credentials and the secret key before deployment.
DEMO_ADMIN_USERNAME = "admin"
DEMO_ADMIN_PASSWORD = "admin123"

LANGUAGES = {
    "en": {
        "app.title": "Diabetes Risk Care",
        "nav.dashboard": "Dashboard",
        "nav.risk_assessment": "Risk Assessment",
        "nav.prediction_history": "Prediction History",
        "nav.logout": "Logout",
        "nav.admin_dashboard": "Dashboard",
        "nav.admin_patients": "Patients",
        "nav.admin_predictions": "Predictions",
        "nav.admin_analytics": "Analytics",
        "nav.admin_reports": "Reports",
        "nav.language": "Language",
        "common.patient_portal": "Patient portal",
        "common.my_workspace": "My workspace",
        "common.patient_dashboard": "Patient dashboard",
        "common.good_to_see": "Good to see you, {name}.",
        "common.personal_workspace_ready": "Your personal health workspace is ready.",
        "common.patient_name": "Patient name",
        "common.age": "Age",
        "common.gender": "Gender",
        "common.previous_predictions": "Previous predictions",
        "common.latest_prediction": "Latest prediction",
        "common.not_available": "Not available",
        "common.next_step": "Next step",
        "common.take_assessment": "Take a risk assessment",
        "common.open_assessment": "Open assessment",
        "common.complete_assessment_hint": "Complete a risk assessment to see your latest result here.",
        "common.assessment_date": "Assessment Date",
        "common.prediction": "Prediction",
        "common.model_confidence": "Model Confidence",
        "common.no_records": "No prediction records yet.",
        "common.records": "records",
        "common.private_assessment": "Private assessment",
        "common.risk_assessment": "Risk assessment",
        "status.high_risk": "High Risk",
        "status.low_risk": "Low Risk",
        "status.high_risk_label": "HIGH RISK",
        "status.low_risk_label": "LOW RISK",
        "auth.sign_in": "Sign in",
        "auth.welcome_back": "Welcome back",
        "auth.new_patient": "New patient?",
        "auth.create_account": "Create account",
        "auth.patient_access": "Patient access",
        "auth.start_private_workspace": "Start your private health workspace.",
        "auth.create_patient_account": "Create a patient account to view your profile and future prediction history.",
        "auth.patient_registration": "Patient registration",
        "auth.already_registered": "Already registered?",
        "auth.register": "Register",
        "auth.confirm_password": "Confirm password",
        "auth.username": "Username",
        "auth.password": "Password",
        "auth.patient_name": "Patient name",
        "auth.enter_health_info": "Enter the health information below to generate a Random Forest risk prediction.",
        "auth.form_note": "New patient?",
        "auth.submit": "Generate risk assessment",
        "auth.upload_report": "Upload Medical Report (Optional)",
        "auth.upload_report_desc": "Upload a medical or laboratory report to automatically extract available health values. Please review all extracted values before generating your risk assessment.",
        "auth.upload_button": "Upload Report & Auto-Fill",
        "auth.file_label": "PDF, JPG, JPEG, or PNG",
        "auth.values_extracted": "Values extracted from this report",
        "auth.report_success": "Report processed successfully. Please review the auto-filled values and complete any missing information before generating your assessment.",
        "auth.report_none": "No relevant medical values were found in the report. Please complete the form manually.",
        "auth.login_required": "Please log in to continue.",
        "auth.invalid_credentials": "Invalid username or password.",
        "auth.admin_reserved": "That username is reserved.",
        "auth.registration_complete": "Registration complete. You can now log in.",
        "auth.username_taken": "That username is already in use.",
        "auth.password_short": "Password must be at least 8 characters.",
        "auth.password_mismatch": "Passwords do not match.",
        "auth.fill_all": "Please complete every field with valid information.",
        "auth.logout_success": "You have been logged out.",
        "auth.access_denied": "You are not authorized to view that page.",
        "auth.out_of_range_age": "Age must be realistic and no greater than 120.",
        "form.age": "Age",
        "form.gender": "Gender",
        "form.gender_select": "Select gender",
        "form.gender_female": "Female",
        "form.gender_male": "Male",
        "form.gender_other": "Other",
        "form.bmi": "BMI",
        "form.pregnancies": "Pregnancies",
        "form.blood_glucose": "Blood glucose level",
        "form.hba1c": "HbA1c level",
        "form.hypertension": "Hypertension",
        "form.heart_disease": "Heart disease",
        "form.smoking_history": "Smoking history",
        "form.select_history": "Select history",
        "form.select_gender": "Select gender",
        "form.never": "Never",
        "form.former": "Former",
        "form.current": "Current",
        "form.yes": "Yes",
        "form.no": "No",
        "form.basic_info": "Basic information",
        "form.clinical_measurements": "Clinical measurements",
        "form.health_history": "Health history",
        "form.disclaimer_note": "This assessment is for educational and research purposes only.",
        "form.select_valid": "Please select a valid smoking history.",
        "result.assessment_complete": "Assessment complete",
        "result.diabetes_risk_result": "Diabetes Risk Assessment Result",
        "result.disclaimer": "This prediction is for educational/research purposes only and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.",
        "result.back_dashboard": "Back to Dashboard",
        "result.view_history": "View Prediction History",
        "result.new_assessment": "New Assessment",
        "result.assessed_on": "Assessed on {date}",
        "admin.control_center": "Control center",
        "admin.system_overview": "System overview",
        "admin.monitor_activity": "Monitor registered patients and prediction activity.",
        "admin.total_patients": "Total patients",
        "admin.total_predictions": "Total predictions",
        "admin.high_risk_predictions": "High risk predictions",
        "admin.low_risk_predictions": "Low risk predictions",
        "admin.high_risk_percentage": "High risk percentage",
        "admin.low_risk_percentage": "Low risk percentage",
        "admin.directory": "Directory",
        "admin.patient_records": "Patient records",
        "admin.activity": "Activity",
        "admin.recent_predictions": "Recent predictions",
        "admin.no_patient_accounts": "No patient accounts registered.",
        "admin.no_prediction_records": "No prediction records yet.",
        "admin.analytics": "Admin analytics",
        "admin.analytics_heading": "Signals at a glance",
        "admin.analytics_desc": "Live summaries calculated from the prediction database.",
        "admin.patient_count": "Patient count",
        "admin.prediction_count": "Prediction count",
        "admin.high_risk": "High risk",
        "admin.low_risk": "Low risk",
        "admin.outcome_distribution": "Outcome distribution",
        "admin.high_risk_vs_low_risk": "High risk vs low risk",
        "admin.prediction_activity": "Prediction activity",
        "admin.recent_activity": "Recent assessment volume",
        "admin.no_activity": "No prediction activity yet.",
        "admin.live_activity": "Live activity",
        "admin.recent_prediction_activity": "Recent prediction activity",
        "admin.records_ready": "Records ready to export",
        "admin.export_patient_records": "Export Patient Records",
        "admin.export_prediction_records": "Export Prediction Records",
        "admin.registered_patients": "Registered patients",
        "admin.assessment_history": "Assessment history",
        "admin.patient_name": "Patient Name",
        "admin.username": "Username",
        "admin.registration_date": "Registration Date",
        "admin.prediction_date": "Prediction Date",
        "admin.download_info": "Download current SQLite records for analysis and reporting.",
        "admin.records_count": "{count} records",
        "validation.age_required": "Age must be greater than 0.",
        "validation.age_integer": "Age must be a whole number.",
        "validation.age_realistic": "Age must be realistic and no greater than 120.",
        "validation.pregnancies_negative": "Pregnancies cannot be negative.",
        "validation.pregnancies_integer": "Pregnancies must be a whole number.",
        "validation.gender_required": "Please select a gender.",
        "validation.hypertension_required": "Hypertension must be 0 or 1.",
        "validation.heart_disease_required": "Heart disease must be 0 or 1.",
        "validation.bmi_positive": "BMI must be greater than 0.",
        "validation.blood_glucose_positive": "Blood glucose level must be greater than 0.",
        "validation.hba1c_positive": "HbA1c level must be greater than 0.",
        "validation.bmi_numeric": "BMI must be a valid number.",
        "validation.blood_glucose_numeric": "Blood glucose level must be a valid number.",
        "validation.hba1c_numeric": "HbA1c level must be a valid number.",
        "validation.smoking_history_invalid": "Please select a valid smoking history.",
        "flash.login_required": "Please log in to continue.",
        "flash.unauthorized": "You are not authorized to view that page.",
        "flash.invalid_credentials": "Invalid username or password.",
        "flash.fill_all": "Please complete every field with valid information.",
        "flash.password_short": "Password must be at least 8 characters.",
        "flash.password_mismatch": "Passwords do not match.",
        "flash.admin_reserved": "That username is reserved.",
        "flash.registration_complete": "Registration complete. You can now log in.",
        "flash.username_taken": "That username is already in use.",
        "flash.logout_success": "You have been logged out.",
        "flash.assessment_service_unavailable": "The risk assessment service is temporarily unavailable.",
        "flash.assessment_failed": "We could not complete this assessment. Please try again.",
        "flash.profile_missing": "Your patient profile could not be found.",
        "flash.upload_missing": "Please choose a PDF, JPG, JPEG, or PNG report.",
        "flash.upload_unsupported": "Unsupported file type. Please upload a PDF, JPG, JPEG, or PNG file.",
        "flash.upload_empty": "The uploaded report is empty.",
        "flash.upload_too_large": "The report is too large. Please upload a file no larger than 10 MB.",
        "flash.report_could_not_read": "The PDF could not be read.",
        "flash.pdf_no_readable_text": "The PDF does not contain readable text.",
        "flash.pdf_extraction_unavailable": "PDF extraction is unavailable. Install the project requirements and restart the application.",
        "flash.image_could_not_read": "The image could not be read.",
        "flash.image_ocr_unavailable": "The image could not be read by the OCR service.",
        "flash.image_no_readable_text": "The image does not contain readable text.",
        "flash.no_relevant_values": "No relevant medical values were found in the report. Please complete the form manually.",
        "flash.invalid_form": "Please complete every field with valid information.",
        "flash.general_error": "We could not complete this assessment. Please try again.",
        "common.language_toggle_en": "English",
        "common.language_toggle_ta": "தமிழ்",
        "common.english": "English",
        "common.tamil": "தமிழ்",
        "common.patient_records": "Patient records",
        "common.prediction_records": "Prediction records",
        "common.from_report": "Values extracted from this report",
        "common.review_extract": "Please review the auto-filled values and complete any missing information before generating your assessment.",
        "common.educational_only": "This assessment is for educational and research purposes only.",
        "common.upload_instructions": "Upload a medical or laboratory report to automatically extract available health values. Please review all extracted values before generating your risk assessment.",
        "common.type_hint": "PDF, JPG, JPEG, or PNG",
        "common.confidence_percent": "{value}%",
        "common.download": "Download",
        "common.no_prediction_history": "No prediction records yet.",
        "common.history": "Prediction history",
        "common.your_records": "Your records",
        "common.assessment_done": "Assessment complete",
        "common.private_workspace": "Private health workspace",
        "common.patient_history": "Prediction History",
        "common.complete_assessment": "Complete a risk assessment to see your latest result here.",
        "common.assessment_coming_next": "Assessment coming next",
        "common.future_workflow": "This area is reserved for the future risk-assessment workflow.",
        "common.medical_disclaimer": "This prediction is for educational/research purposes only and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.",
    },
    "ta": {
        "app.title": "டயாபிட்டிஸ் ரிஸ்க் கேரே",
        "nav.dashboard": "டாஷ்போர்டு",
        "nav.risk_assessment": "ஆய்வு மதிப்பீடு",
        "nav.prediction_history": "முன்கணிப்பு வரலாறு",
        "nav.logout": "வெளியேறு",
        "nav.admin_dashboard": "டாஷ்போர்டு",
        "nav.admin_patients": "நோயாளிகள்",
        "nav.admin_predictions": "முன்கணிப்புகள்",
        "nav.admin_analytics": "விவரங்கள்",
        "nav.admin_reports": "அறிக்கைகள்",
        "nav.language": "மொழி",
        "common.patient_portal": "நோயாளர் போர்டல்",
        "common.my_workspace": "என் பணியிடம்",
        "common.patient_dashboard": "நோயாளர் டாஷ்போர்டு",
        "common.good_to_see": "உங்களைப் பார்க்க மகிழ்ச்சி, {name}.",
        "common.personal_workspace_ready": "உங்கள் தனிப்பட்ட உடல்நலம் பணியிடம் தயார்.",
        "common.patient_name": "நோயாளியின் பெயர்",
        "common.age": "வயது",
        "common.gender": "பாலினம்",
        "common.previous_predictions": "முந்தைய முன்கணிப்புகள்",
        "common.latest_prediction": "சமீபத்திய முன்கணிப்பு",
        "common.not_available": "கிடைக்கவில்லை",
        "common.next_step": "அடுத்த படி",
        "common.take_assessment": "ஆய்வு மதிப்பீட்டை எடுத்துக்கொள்ளுங்கள்",
        "common.open_assessment": "மதிப்பீட்டை திறக்கவும்",
        "common.complete_assessment_hint": "சமீபத்திய முடிவைப் பார்க்க ஒரு ரிஸ்க் மதிப்பீட்டை முடிக்கவும்.",
        "common.assessment_date": "மதிப்பீடு தேதி",
        "common.prediction": "முன்கணிப்பு",
        "common.model_confidence": "மாடல் நம்பிக்கை",
        "common.no_records": "இன்னும் முன்கணிப்பு பதிவுகள் இல்லை.",
        "common.records": "பதிவுகள்",
        "common.private_assessment": "தனிப்பட்ட மதிப்பீடு",
        "common.risk_assessment": "ஆய்வு மதிப்பீடு",
        "status.high_risk": "அதிக ஆபத்து",
        "status.low_risk": "குறைந்த ஆபத்து",
        "status.high_risk_label": "அதிக ஆபத்து",
        "status.low_risk_label": "குறைந்த ஆபத்து",
        "auth.sign_in": "உள்நுழை",
        "auth.welcome_back": "மீண்டும் வருக",
        "auth.new_patient": "புதிய நோயாளியா?",
        "auth.create_account": "கணக்கு உருவாக்கு",
        "auth.patient_access": "நோயாளர் அணுகல்",
        "auth.start_private_workspace": "உங்கள் தனிப்பட்ட உடல்நல பணியிடத்தை தொடங்குங்கள்.",
        "auth.create_patient_account": "உங்கள் சுயவிவரம் மற்றும் எதிர்கால முன்கணிப்பு வரலாற்றைப் பார்க்க நோயாளர் கணக்கை உருவாக்கவும்.",
        "auth.patient_registration": "நோயாளர் பதிவு",
        "auth.already_registered": "ஏற்கனவே பதிவு செய்யப்பட்டிருக்கிறதா?",
        "auth.register": "பதிவு செய்",
        "auth.confirm_password": "கடவுச்சொல்லை உறுதிப்படுத்து",
        "auth.username": "பயனர்பெயர்",
        "auth.password": "கடவுச்சொல்",
        "auth.patient_name": "நோயாளியின் பெயர்",
        "auth.enter_health_info": "ரேண்டம் ஃபாரெஸ்ட் ரிஸ்க் முன்கணிப்பை உருவாக்க கீழே உடல்நலத் தகவலை உள்ளிடவும்.",
        "auth.form_note": "புதிய நோயாளியா?",
        "auth.submit": "ரிஸ்க் மதிப்பீட்டை உருவாக்கு",
        "auth.upload_report": "மருத்துவ அறிக்கையை பதிவேற்று (விருப்பம்)",
        "auth.upload_report_desc": "கிடைக்கக்கூடிய உடல்நல மதிப்புகளை தானாகப் பிரித்தெடுக்க மருத்துவ அல்லது ஆய்வக அறிக்கையை பதிவேற்றவும். உங்கள் மதிப்பீட்டை உருவாக்குவதற்கு முன் பிரித்தெடுக்கப்பட்ட அனைத்து மதிப்புகளையும் பரிசீலிக்கவும்.",
        "auth.upload_button": "அறிக்கையை பதிவேற்று & தானாக நிரப்பவும்",
        "auth.file_label": "PDF, JPG, JPEG, அல்லது PNG",
        "auth.values_extracted": "இந்த அறிக்கையில் பிரித்தெடுக்கப்பட்ட மதிப்புகள்",
        "auth.report_success": "அறிக்கை செயலாக்கப்பட்டது. தானாக நிரப்பப்பட்ட மதிப்புகளை மதிப்பாய்வு செய்து, உங்கள் மதிப்பீட்டை உருவாக்குவதற்கு முன் எந்தவொரு விடுபட்ட தகவலையும் முடிக்கவும்.",
        "auth.report_none": "அறிக்கையில் தொடர்புடைய மருத்துவ மதிப்புகள் எதுவும் காணப்படவில்லை. தயவுசெய்து கைமுறையாகப் பூர்த்தி செய்யவும்.",
        "auth.login_required": "தொடருவதற்கு முதலில் உள்நுழையவும்.",
        "auth.invalid_credentials": "பயனர்பெயர் அல்லது கடவுச்சொல் தவறானது.",
        "auth.admin_reserved": "அந்த பயனர்பெயர் ஒதுக்கப்பட்டுள்ளது.",
        "auth.registration_complete": "பதிவு முடிந்தது. இப்போது உள்நுழையலாம்.",
        "auth.username_taken": "அந்த பயனர்பெயர் ஏற்கனவே பயன்பாட்டில் உள்ளது.",
        "auth.password_short": "கடவுச்சொல் குறைந்தபட்சம் 8 எழுத்துகளாக இருக்க வேண்டும்.",
        "auth.password_mismatch": "கடவுச்சொற்கள் பொருந்தவில்லை.",
        "auth.fill_all": "சரியான தகவலுடன் எல்லா புலங்களையும் பூர்த்தி செய்யவும்.",
        "auth.logout_success": "நீங்கள் வெளியேறிவிட்டீர்கள்.",
        "auth.access_denied": "இந்தப் பக்கத்தை நீங்கள் பார்க்க அனுமதிக்கப்படவில்லை.",
        "auth.out_of_range_age": "வயது 120 ஐ விட அதிகமாக இருக்க முடியாது.",
        "form.age": "வயது",
        "form.gender": "பாலினம்",
        "form.gender_select": "பாலினத்தை தேர்ந்தெடுக்கவும்",
        "form.gender_female": "பெண்",
        "form.gender_male": "ஆண்",
        "form.gender_other": "மற்றவை",
        "form.bmi": "BMI",
        "form.pregnancies": "கர்ப்பங்கள்",
        "form.blood_glucose": "இரத்த சர்க்கரை நிலை",
        "form.hba1c": "HbA1c நிலை",
        "form.hypertension": "உயர் இரத்த அழுத்தம்",
        "form.heart_disease": "இதய நோய்",
        "form.smoking_history": "புகைபிடிக்கும் பழக்கம்",
        "form.select_history": "வரலாற்றை தேர்ந்தெடுக்கவும்",
        "form.select_gender": "பாலினத்தை தேர்ந்தெடுக்கவும்",
        "form.never": "ஒருபோதும் இல்லை",
        "form.former": "முன்பு",
        "form.current": "தற்போது",
        "form.yes": "ஆம்",
        "form.no": "இல்லை",
        "form.basic_info": "அடிப்படை தகவல்",
        "form.clinical_measurements": "மருத்துவ அளவீடுகள்",
        "form.health_history": "சுகாதார வரலாறு",
        "form.disclaimer_note": "இந்த மதிப்பீடு கல்வி மற்றும் ஆராய்ச்சி நோக்கங்களுக்காக மட்டுமே.",
        "form.select_valid": "சரியான புகைபிடிக்கும் பழக்கத்தை தேர்ந்தெடுக்கவும்.",
        "result.assessment_complete": "மதிப்பீடு முடிந்தது",
        "result.diabetes_risk_result": "டயாபிட்டிஸ் ரிஸ்க் மதிப்பீட்டு முடிவு",
        "result.disclaimer": "இந்த முன்கணிப்பு கல்வி/ஆராய்ச்சி நோக்கங்களுக்காக மட்டுமே, மருத்துவ நோயறிதல் அல்ல. மருத்துவ ஆலோசனைக்கு தகுதியான சுகாதார நிபுணரை அணுகவும்.",
        "result.back_dashboard": "டாஷ்போர்டுக்குத் திரும்பு",
        "result.view_history": "முன்கணிப்பு வரலாற்றை பார்க்கவும்",
        "result.new_assessment": "புதிய மதிப்பீடு",
        "result.assessed_on": "மதிப்பீடு செய்யப்பட்ட நாள்: {date}",
        "admin.control_center": "கட்டுப்பாட்டு மையம்",
        "admin.system_overview": "சிஸ்டம் மேலோட்டம்",
        "admin.monitor_activity": "பதிவுசெய்யப்பட்ட நோயாளிகள் மற்றும் முன்கணிப்பு செயல்பாட்டைக் கண்காணிக்கவும்.",
        "admin.total_patients": "மொத்த நோயாளிகள்",
        "admin.total_predictions": "மொத்த முன்கணிப்புகள்",
        "admin.high_risk_predictions": "அதிக ஆபத்து முன்கணிப்புகள்",
        "admin.low_risk_predictions": "குறைந்த ஆபத்து முன்கணிப்புகள்",
        "admin.high_risk_percentage": "அதிக ஆபத்து சதவீதம்",
        "admin.low_risk_percentage": "குறைந்த ஆபத்து சதவீதம்",
        "admin.directory": "அடைவு",
        "admin.patient_records": "நோயாளி பதிவுகள்",
        "admin.activity": "செயல்பாடு",
        "admin.recent_predictions": "சமீபத்திய முன்கணிப்புகள்",
        "admin.no_patient_accounts": "எந்த நோயாளி கணக்குகளும் பதிவு செய்யப்படவில்லை.",
        "admin.no_prediction_records": "இன்னும் முன்கணிப்பு பதிவுகள் இல்லை.",
        "admin.analytics": "நிர்வாக பகுப்பாய்வு",
        "admin.analytics_heading": "ஒரே பார்வையில் சமிக்ஞைகள்",
        "admin.analytics_desc": "முன்கணிப்பு தரவுத்தளத்திலிருந்து கணக்கிடப்பட்ட நேரடி சுருக்கங்கள்.",
        "admin.patient_count": "நோயாளி எண்ணிக்கை",
        "admin.prediction_count": "முன்கணிப்பு எண்ணிக்கை",
        "admin.high_risk": "அதிக ஆபத்து",
        "admin.low_risk": "குறைந்த ஆபத்து",
        "admin.outcome_distribution": "முடிவு விநியோகம்",
        "admin.high_risk_vs_low_risk": "அதிக ஆபத்து vs குறைந்த ஆபத்து",
        "admin.prediction_activity": "முன்கணிப்பு செயல்பாடு",
        "admin.recent_activity": "சமீபத்திய மதிப்பீட்டு அளவு",
        "admin.no_activity": "இன்னும் முன்கணிப்பு செயல்பாடு இல்லை.",
        "admin.live_activity": "நேரடி செயல்பாடு",
        "admin.recent_prediction_activity": "சமீபத்திய முன்கணிப்பு செயல்பாடு",
        "admin.records_ready": "ஏற்றுமதி செய்யத் தயாரான பதிவுகள்",
        "admin.export_patient_records": "நோயாளி பதிவுகளை ஏற்றுமதி செய்",
        "admin.export_prediction_records": "முன்கணிப்பு பதிவுகளை ஏற்றுமதி செய்",
        "admin.registered_patients": "பதிவுசெய்யப்பட்ட நோயாளிகள்",
        "admin.assessment_history": "மதிப்பீட்டு வரலாறு",
        "admin.patient_name": "நோயாளியின் பெயர்",
        "admin.username": "பயனர்பெயர்",
        "admin.registration_date": "பதிவு நாள்",
        "admin.prediction_date": "முன்கணிப்பு தேதி",
        "admin.download_info": "பகுப்பாய்வு மற்றும் அறிக்கைக்காக தற்போதைய SQLite பதிவுகளைப் பதிவிறக்கவும்.",
        "admin.records_count": "{count} பதிவுகள்",
        "validation.age_required": "வயது 0 விட அதிகமாக இருக்க வேண்டும்.",
        "validation.age_integer": "வயது முழு எண்ணாக இருக்க வேண்டும்.",
        "validation.age_realistic": "வயது உண்மையானதாக இருக்க வேண்டும் மற்றும் 120 ஐ விட அதிகமாக இருக்கக்கூடாது.",
        "validation.pregnancies_negative": "கர்ப்பங்கள் எதிர்மறையாக இருக்க முடியாது.",
        "validation.pregnancies_integer": "கர்ப்பங்கள் முழு எண்ணாக இருக்க வேண்டும்.",
        "validation.gender_required": "பாலினத்தைத் தேர்ந்தெடுக்கவும்.",
        "validation.hypertension_required": "உயர் இரத்த அழுத்தம் 0 அல்லது 1 ஆக இருக்க வேண்டும்.",
        "validation.heart_disease_required": "இதய நோய் 0 அல்லது 1 ஆக இருக்க வேண்டும்.",
        "validation.bmi_positive": "BMI 0 விட அதிகமாக இருக்க வேண்டும்.",
        "validation.blood_glucose_positive": "இரத்த சர்க்கரை நிலை 0 விட அதிகமாக இருக்க வேண்டும்.",
        "validation.hba1c_positive": "HbA1c நிலை 0 விட அதிகமாக இருக்க வேண்டும்.",
        "validation.bmi_numeric": "BMI சரியான எண்ணாக இருக்க வேண்டும்.",
        "validation.blood_glucose_numeric": "இரத்த சர்க்கரை நிலை சரியான எண்ணாக இருக்க வேண்டும்.",
        "validation.hba1c_numeric": "HbA1c நிலை சரியான எண்ணாக இருக்க வேண்டும்.",
        "validation.smoking_history_invalid": "சரியான புகைபிடிக்கும் பழக்கத்தை தேர்ந்தெடுக்கவும்.",
        "flash.login_required": "தொடருவதற்கு முதலில் உள்நுழையவும்.",
        "flash.unauthorized": "இந்தப் பக்கத்தை நீங்கள் பார்க்க அனுமதிக்கப்படவில்லை.",
        "flash.invalid_credentials": "பயனர்பெயர் அல்லது கடவுச்சொல் தவறானது.",
        "flash.fill_all": "சரியான தகவலுடன் எல்லா புலங்களையும் பூர்த்தி செய்யவும்.",
        "flash.password_short": "கடவுச்சொல் குறைந்தபட்சம் 8 எழுத்துகளாக இருக்க வேண்டும்.",
        "flash.password_mismatch": "கடவுச்சொற்கள் பொருந்தவில்லை.",
        "flash.admin_reserved": "அந்த பயனர்பெயர் ஒதுக்கப்பட்டுள்ளது.",
        "flash.registration_complete": "பதிவு முடிந்தது. இப்போது உள்நுழையலாம்.",
        "flash.username_taken": "அந்த பயனர்பெயர் ஏற்கனவே பயன்பாட்டில் உள்ளது.",
        "flash.logout_success": "நீங்கள் வெளியேறிவிட்டீர்கள்.",
        "flash.assessment_service_unavailable": "ரிஸ்க் மதிப்பீட்டு சேவை தற்காலிகமாக கிடைக்கவில்லை.",
        "flash.assessment_failed": "இந்த மதிப்பீட்டை முடிக்க முடியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
        "flash.profile_missing": "உங்கள் நோயாளி சுயவிவரம் காணப்படவில்லை.",
        "flash.upload_missing": "PDF, JPG, JPEG அல்லது PNG அறிக்கையைத் தேர்ந்தெடுக்கவும்.",
        "flash.upload_unsupported": "ஆதரிக்கப்படாத கோப்பு வகை. PDF, JPG, JPEG, அல்லது PNG கோப்பை பதிவேற்றவும்.",
        "flash.upload_empty": "பதிவேற்றப்பட்ட அறிக்கை காலியாக உள்ளது.",
        "flash.upload_too_large": "அறிக்கை மிகப்பெரியது. 10 MB க்கும் குறைவான கோப்பை பதிவேற்றவும்.",
        "flash.report_could_not_read": "PDF ஐப் படிக்க முடியவில்லை.",
        "flash.pdf_no_readable_text": "PDF இல் படிக்கக்கூடிய உரை இல்லை.",
        "flash.pdf_extraction_unavailable": "PDF பிரித்தெடுப்பு சேவை கிடைக்கவில்லை. திட்டத்தின் தேவையான தொகுப்புகளை நிறுவி செயலியை மீண்டும் தொடங்கவும்.",
        "flash.image_could_not_read": "படத்தைப் படிக்க முடியவில்லை.",
        "flash.image_ocr_unavailable": "OCR சேவையால் படத்தைப் படிக்க முடியவில்லை.",
        "flash.image_no_readable_text": "படத்தில் படிக்கக்கூடிய உரை இல்லை.",
        "flash.no_relevant_values": "அறிக்கையில் தொடர்புடைய மருத்துவ மதிப்புகள் எதுவும் காணப்படவில்லை. தயவுசெய்து கைமுறையாக பூர்த்தி செய்யவும்.",
        "flash.invalid_form": "சரியான தகவலுடன் எல்லா புலங்களையும் பூர்த்தி செய்யவும்.",
        "flash.general_error": "இந்த மதிப்பீட்டை முடிக்க முடியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
        "common.language_toggle_en": "English",
        "common.language_toggle_ta": "தமிழ்",
        "common.english": "English",
        "common.tamil": "தமிழ்",
        "common.patient_records": "நோயாளி பதிவுகள்",
        "common.prediction_records": "முன்கணிப்பு பதிவுகள்",
        "common.from_report": "இந்த அறிக்கையில் பிரித்தெடுக்கப்பட்ட மதிப்புகள்",
        "common.review_extract": "தானாக நிறைவு செய்யப்பட்ட மதிப்புகளை மதிப்பாய்வு செய்து, உங்கள் மதிப்பீட்டை உருவாக்குவதற்கு முன் எந்த விடுபட்ட தகவலையும் பூர்த்தி செய்யவும்.",
        "common.educational_only": "இந்த மதிப்பீடு கல்வி மற்றும் ஆராய்ச்சி நோக்கங்களுக்காக மட்டுமே.",
        "common.upload_instructions": "கிடைக்கக்கூடிய உடல்நல மதிப்புகளை தானாகப் பிரித்தெடுக்க மருத்துவ அல்லது ஆய்வக அறிக்கையை பதிவேற்றவும். உங்கள் மதிப்பீட்டை உருவாக்குவதற்கு முன் பிரித்தெடுக்கப்பட்ட அனைத்து மதிப்புகளையும் பரிசீலிக்கவும்.",
        "common.type_hint": "PDF, JPG, JPEG, அல்லது PNG",
        "common.confidence_percent": "{value}%",
        "common.download": "பதிவிறக்கு",
        "common.no_prediction_history": "இன்னும் முன்கணிப்பு பதிவுகள் இல்லை.",
        "common.history": "முன்கணிப்பு வரலாறு",
        "common.your_records": "உங்கள் பதிவுகள்",
        "common.assessment_done": "மதிப்பீடு முடிந்தது",
        "common.private_workspace": "தனிப்பட்ட உடல்நல பணியிடம்",
        "common.patient_history": "முன்கணிப்பு வரலாறு",
        "common.complete_assessment": "சமீபத்திய முடிவைப் பார்க்க ஒரு ரிஸ்க் மதிப்பீட்டை முடிக்கவும்.",
        "common.assessment_coming_next": "அடுத்த மதிப்பீடு",
        "common.future_workflow": "எதிர்கால ரிஸ்க் மதிப்பீட்டு பணிப்பாய்வு இதற்கு ஒதுக்கப்பட்டுள்ளது.",
        "common.medical_disclaimer": "இந்த முன்கணிப்பு கல்வி/ஆராய்ச்சி நோக்கங்களுக்காக மட்டுமே, மருத்துவ நோயறிதல் அல்ல. மருத்துவ ஆலோசனைக்கு தகுதியான சுகாதார நிபுணரை அணுகவும்.",
    },
}


def get_language():
    language = session.get("language", "en")
    return language if language in LANGUAGES else "en"


def translate(key, **kwargs):
    language = get_language()
    template = LANGUAGES.get(language, LANGUAGES["en"]).get(key, LANGUAGES["en"].get(key, key))
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return template
    return template


@app.before_request
def set_default_language():
    if "language" not in session or session.get("language") not in LANGUAGES:
        session["language"] = "en"


@app.context_processor
def inject_locale():
    return {
        "current_user": session.get("username"),
        "current_role": session.get("role"),
        "current_lang": get_language(),
        "t": translate,
        "language_options": LANGUAGES,
        "lang_en": "en",
        "lang_ta": "ta",
    }


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_error):
    database = g.pop("db", None)
    if database is not None:
        database.close()


def init_db():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(app.config["DATABASE"])
    database.execute("PRAGMA foreign_keys = ON")
    database.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('patient', 'admin'))
        );

        CREATE TABLE IF NOT EXISTS patient_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            patient_name TEXT NOT NULL,
            age INTEGER NOT NULL CHECK (age > 0),
            gender TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
            probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
            prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patient_profiles (id) ON DELETE CASCADE
        );
        """
    )
    admin = database.execute(
        "SELECT id FROM users WHERE username = ?", (DEMO_ADMIN_USERNAME,)
    ).fetchone()
    if admin is None:
        database.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (
                DEMO_ADMIN_USERNAME,
                generate_password_hash(DEMO_ADMIN_PASSWORD),
                "admin",
            ),
        )
    database.commit()
    database.close()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash(translate("flash.login_required"), "info")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def role_required(role):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped_view(*args, **kwargs):
            if session.get("role") != role:
                flash(translate("flash.unauthorized"), "error")
                destination = "admin_dashboard" if session.get("role") == "admin" else "patient_dashboard"
                return redirect(url_for(destination))
            return view(*args, **kwargs)

        return wrapped_view

    return decorator


def validate_assessment_form(form):
    errors = []
    values = {
        "age": form.get("age", "").strip(),
        "gender": form.get("gender", "").strip(),
        "bmi": form.get("bmi", "").strip(),
        "blood_glucose_level": form.get("blood_glucose_level", "").strip(),
        "hba1c_level": form.get("hba1c_level", "").strip(),
        "hypertension": form.get("hypertension", "").strip(),
        "heart_disease": form.get("heart_disease", "").strip(),
        "smoking_history": form.get("smoking_history", "").strip().lower(),
        "pregnancies": form.get("pregnancies", "").strip(),
    }

    numeric_fields = {
        "bmi": "validation.bmi_positive",
        "blood_glucose_level": "validation.blood_glucose_positive",
        "hba1c_level": "validation.hba1c_positive",
    }
    numeric_values = {}
    for field, key in numeric_fields.items():
        try:
            numeric_values[field] = float(values[field])
            if not math.isfinite(numeric_values[field]) or numeric_values[field] <= 0:
                errors.append(translate(key))
        except ValueError:
            if field == "bmi":
                errors.append(translate("validation.bmi_numeric"))
            elif field == "blood_glucose_level":
                errors.append(translate("validation.blood_glucose_numeric"))
            else:
                errors.append(translate("validation.hba1c_numeric"))

    try:
        numeric_values["age"] = int(values["age"])
        if numeric_values["age"] <= 0:
            errors.append(translate("validation.age_required"))
    except ValueError:
        errors.append(translate("validation.age_integer"))

    if "age" in numeric_values and numeric_values["age"] > 120:
        errors.append(translate("validation.age_realistic"))

    try:
        numeric_values["pregnancies"] = int(values["pregnancies"])
        if numeric_values["pregnancies"] < 0:
            errors.append(translate("validation.pregnancies_negative"))
    except ValueError:
        errors.append(translate("validation.pregnancies_integer"))

    if not values["gender"]:
        errors.append(translate("validation.gender_required"))
    if values["hypertension"] not in {"0", "1"}:
        errors.append(translate("validation.hypertension_required"))
    if values["heart_disease"] not in {"0", "1"}:
        errors.append(translate("validation.heart_disease_required"))
    if values["smoking_history"] not in {"current", "former", "never"}:
        errors.append(translate("validation.smoking_history_invalid"))

    if errors:
        return values, errors

    numeric_values["hypertension"] = int(values["hypertension"])
    numeric_values["heart_disease"] = int(values["heart_disease"])
    numeric_values["gender"] = values["gender"]
    numeric_values["smoking_history"] = values["smoking_history"]
    return numeric_values, errors


def create_model_input(values):
    selected_df = pd.read_csv(SELECTED_FEATURES_PATH)
    selected_feature_names = [
        column for column in selected_df.columns if column != "Diabetes_Outcome"
    ]
    patient_data = pd.DataFrame(
        [
            {
                "Age": values["age"],
                "BMI": values["bmi"],
                "Blood_Glucose_Level": values["blood_glucose_level"],
                "HbA1c_Level": values["hba1c_level"],
                "Hypertension": values["hypertension"],
                "Heart_Disease": values["heart_disease"],
                "Smoking_History": values["smoking_history"],
            }
        ]
    )

    patient_data["BMI_Category"] = pd.cut(
        patient_data["BMI"],
        bins=[0, 18.5, 24.9, 29.9, float("inf")],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,
    )
    patient_data["Age_Group"] = pd.cut(
        patient_data["Age"],
        bins=[0, 18, 35, 50, 65, float("inf")],
        labels=["0-17", "18-34", "35-49", "50-64", "65+"],
        right=False,
    )
    patient_data["Glucose_Risk"] = pd.cut(
        patient_data["Blood_Glucose_Level"],
        bins=[0, 99, 125, float("inf")],
        labels=["Normal", "Prediabetes", "High_Risk"],
        right=False,
    )
    patient_data["HbA1c_Risk"] = pd.cut(
        patient_data["HbA1c_Level"],
        bins=[0, 5.7, 6.4, float("inf")],
        labels=["Normal", "Prediabetes", "High_Risk"],
        right=False,
    )
    patient_data["Cardiovascular_Risk"] = (
        patient_data["Hypertension"] + patient_data["Heart_Disease"]
    ).clip(lower=0, upper=2).map({0: "Low", 1: "Moderate", 2: "High"})

    engineered_input = pd.get_dummies(
        patient_data,
        columns=[
            "BMI_Category",
            "Age_Group",
            "Glucose_Risk",
            "HbA1c_Risk",
            "Cardiovascular_Risk",
            "Smoking_History",
        ],
        dtype=int,
    )
    return engineered_input.reindex(columns=selected_feature_names, fill_value=0)


@app.route("/set-language/<lang>")
def set_language(lang):
    if lang not in LANGUAGES:
        lang = "en"
    session["language"] = lang
    next_url = request.args.get("next")
    if next_url and next_url.startswith("/"):
        return redirect(next_url)
    return redirect(url_for("index"))


@app.route("/")
def index():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    if session.get("role") == "patient":
        return redirect(url_for("patient_dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_db().execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if user is None or not check_password_hash(user["password"], password):
            flash(translate("flash.invalid_credentials"), "error")
            return render_template("login.html")
        session.clear()
        session["language"] = get_language()
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]
        return redirect(
            url_for("admin_dashboard" if user["role"] == "admin" else "patient_dashboard")
        )
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        patient_name = request.form.get("patient_name", "").strip()
        gender = request.form.get("gender", "").strip()
        age_text = request.form.get("age", "").strip()

        try:
            age = int(age_text)
        except ValueError:
            age = 0

        if not all([username, password, patient_name, gender]) or age <= 0:
            flash(translate("flash.fill_all"), "error")
        elif len(password) < 8:
            flash(translate("flash.password_short"), "error")
        elif password != confirm_password:
            flash(translate("flash.password_mismatch"), "error")
        elif username.lower() == DEMO_ADMIN_USERNAME:
            flash(translate("flash.admin_reserved"), "error")
        else:
            database = get_db()
            try:
                cursor = database.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), "patient"),
                )
                database.execute(
                    """
                    INSERT INTO patient_profiles (user_id, patient_name, age, gender)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cursor.lastrowid, patient_name, age, gender),
                )
                database.commit()
                flash(translate("flash.registration_complete"), "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                database.rollback()
                flash(translate("flash.username_taken"), "error")
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash(translate("flash.logout_success"), "success")
    return redirect(url_for("login"))


@app.route("/patient")
@app.route("/patient/dashboard")
@role_required("patient")
def patient_dashboard():
    database = get_db()
    profile = database.execute(
        "SELECT * FROM patient_profiles WHERE user_id = ?", (session["user_id"],)
    ).fetchone()
    prediction_count = database.execute(
        "SELECT COUNT(*) AS total FROM prediction_records WHERE patient_id = ?",
        (profile["id"],),
    ).fetchone()["total"]
    latest_prediction = database.execute(
        """
        SELECT * FROM prediction_records
        WHERE patient_id = ?
        ORDER BY prediction_date DESC, id DESC
        LIMIT 1
        """,
        (profile["id"],),
    ).fetchone()
    return render_template(
        "patient_dashboard.html",
        profile=profile,
        prediction_count=prediction_count,
        latest_prediction=latest_prediction,
    )


@app.route("/risk-assessment", methods=["GET", "POST"])
@app.route("/patient/risk-assessment", methods=["GET", "POST"])
@role_required("patient")
def risk_assessment():
    if request.method == "GET":
        return render_template("risk_assessment.html", form_data={})

    values, errors = validate_assessment_form(request.form)
    if errors:
        for error in errors:
            flash(error, "error")
        return render_template("risk_assessment.html", form_data=request.form), 400

    try:
        rf_model = joblib.load(MODEL_PATH)
        input_data = create_model_input(values)
        model_feature_names = list(rf_model.feature_names_in_)
        if list(input_data.columns) != model_feature_names:
            raise ValueError("Prediction input does not match the trained model schema.")

        prediction = int(rf_model.predict(input_data)[0])
        probabilities = rf_model.predict_proba(input_data)[0]
        class_index = list(rf_model.classes_).index(prediction)
        probability = float(probabilities[class_index])
    except FileNotFoundError:
        app.logger.exception("The Random Forest model file could not be found.")
        flash(translate("flash.assessment_service_unavailable"), "error")
        return render_template("risk_assessment.html", form_data=request.form), 500
    except Exception:
        app.logger.exception("Risk prediction failed.")
        flash(translate("flash.assessment_failed"), "error")
        return render_template("risk_assessment.html", form_data=request.form), 500

    database = get_db()
    profile = database.execute(
        "SELECT id FROM patient_profiles WHERE user_id = ?", (session["user_id"],)
    ).fetchone()
    if profile is None:
        flash(translate("flash.profile_missing"), "error")
        return redirect(url_for("patient_dashboard"))

    cursor = database.execute(
        """
        INSERT INTO prediction_records (patient_id, prediction, probability)
        VALUES (?, ?, ?)
        """,
        (profile["id"], prediction, probability),
    )
    database.commit()
    prediction_record = database.execute(
        "SELECT * FROM prediction_records WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    return render_template("prediction_result.html", prediction=prediction_record)


@app.route("/risk-assessment/upload-report", methods=["POST"])
@app.route("/patient/risk-assessment/upload-report", methods=["POST"])
@role_required("patient")
def upload_report():
    report = request.files.get("medical_report")
    if report is None or not report.filename:
        flash(translate("flash.upload_missing"), "error")
        return render_template("risk_assessment.html", form_data={}), 400
    if not is_allowed_filename(report.filename):
        flash(translate("flash.upload_unsupported"), "error")
        return render_template("risk_assessment.html", form_data={}), 400

    report_bytes = report.read(MAX_REPORT_BYTES + 1)
    if not report_bytes:
        flash(translate("flash.upload_empty"), "error")
        return render_template("risk_assessment.html", form_data={}), 400
    if len(report_bytes) > MAX_REPORT_BYTES:
        flash(translate("flash.upload_too_large"), "error")
        return render_template("risk_assessment.html", form_data={}), 413

    try:
        report_text = extract_report_text(report.filename, report_bytes)
        extracted_values = extract_medical_values(report_text)
    except ReportProcessingError as error:
        report_error_keys = {
            "The PDF could not be read.": "flash.report_could_not_read",
            "The PDF does not contain readable text.": "flash.pdf_no_readable_text",
            "PDF extraction is unavailable. Install the project requirements and restart the application.": "flash.pdf_extraction_unavailable",
            "The image could not be read.": "flash.image_could_not_read",
            "The image could not be read by the OCR service.": "flash.image_ocr_unavailable",
            "The image does not contain readable text.": "flash.image_no_readable_text",
        }
        flash(translate(report_error_keys.get(str(error), "flash.report_could_not_read")), "error")
        return render_template("risk_assessment.html", form_data={}), 400

    if not extracted_values:
        flash(translate("flash.no_relevant_values"), "error")
        return render_template("risk_assessment.html", form_data={}), 400

    form_data = {field: str(value) for field, value in extracted_values.items()}
    flash(translate("auth.report_success"), "success")
    return render_template(
        "risk_assessment.html",
        form_data=form_data,
        extracted_values=extracted_values,
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_large_report(_error):
    flash(translate("flash.upload_too_large"), "error")
    return render_template("risk_assessment.html", form_data={}), 413


@app.route("/patient/history")
@role_required("patient")
def patient_history():
    database = get_db()
    profile = database.execute(
        "SELECT * FROM patient_profiles WHERE user_id = ?", (session["user_id"],)
    ).fetchone()
    predictions = database.execute(
        """
        SELECT * FROM prediction_records
        WHERE patient_id = ?
        ORDER BY prediction_date DESC, id DESC
        """,
        (profile["id"],),
    ).fetchall()
    return render_template(
        "patient_dashboard.html", profile=profile, predictions=predictions, history=True
    )


@app.route("/admin")
@app.route("/admin/dashboard")
@role_required("admin")
def admin_dashboard():
    database = get_db()
    total_patients = database.execute(
        "SELECT COUNT(*) AS total FROM patient_profiles"
    ).fetchone()["total"]
    total_predictions = database.execute(
        "SELECT COUNT(*) AS total FROM prediction_records"
    ).fetchone()["total"]
    high_risk = database.execute(
        "SELECT COUNT(*) AS total FROM prediction_records WHERE prediction = 1"
    ).fetchone()["total"]
    low_risk = database.execute(
        "SELECT COUNT(*) AS total FROM prediction_records WHERE prediction = 0"
    ).fetchone()["total"]
    stats = {
        "patients": total_patients,
        "predictions": total_predictions,
        "high_risk": high_risk,
        "low_risk": low_risk,
        "high_risk_percentage": (high_risk / total_predictions * 100) if total_predictions else 0,
        "low_risk_percentage": (low_risk / total_predictions * 100) if total_predictions else 0,
    }
    patients = database.execute(
        """
        SELECT p.*, u.username
        FROM patient_profiles AS p
        JOIN users AS u ON u.id = p.user_id
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    predictions = database.execute(
        """
        SELECT r.*, p.patient_name, u.username
        FROM prediction_records AS r
        JOIN patient_profiles AS p ON p.id = r.patient_id
        JOIN users AS u ON u.id = p.user_id
        ORDER BY r.prediction_date DESC, r.id DESC
        LIMIT 20
        """
    ).fetchall()
    return render_template(
        "admin_dashboard.html", stats=stats, patients=patients, predictions=predictions
    )


@app.route("/admin/patients")
@role_required("admin")
def admin_patients():
    return admin_dashboard()


@app.route("/admin/predictions")
@role_required("admin")
def admin_predictions():
    return admin_dashboard()


@app.route("/admin/analytics")
@role_required("admin")
def admin_analytics():
    database = get_db()
    total_patients = database.execute(
        "SELECT COUNT(*) AS total FROM patient_profiles"
    ).fetchone()["total"]
    total_predictions = database.execute(
        "SELECT COUNT(*) AS total FROM prediction_records"
    ).fetchone()["total"]
    high_risk = database.execute(
        "SELECT COUNT(*) AS total FROM prediction_records WHERE prediction = 1"
    ).fetchone()["total"]
    low_risk = database.execute(
        "SELECT COUNT(*) AS total FROM prediction_records WHERE prediction = 0"
    ).fetchone()["total"]
    activity_rows = database.execute(
        """
        SELECT DATE(prediction_date) AS activity_date, COUNT(*) AS count
        FROM prediction_records
        GROUP BY DATE(prediction_date)
        ORDER BY activity_date DESC
        LIMIT 14
        """
    ).fetchall()
    maximum_activity = max((row["count"] for row in activity_rows), default=0)
    activity = [
        {
            "date": row["activity_date"],
            "count": row["count"],
            "width": (row["count"] / maximum_activity * 100) if maximum_activity else 0,
        }
        for row in activity_rows
    ]
    recent_predictions = database.execute(
        """
        SELECT r.*, p.patient_name, u.username
        FROM prediction_records AS r
        JOIN patient_profiles AS p ON p.id = r.patient_id
        JOIN users AS u ON u.id = p.user_id
        ORDER BY r.prediction_date DESC, r.id DESC
        LIMIT 10
        """
    ).fetchall()
    return render_template(
        "admin_analytics.html",
        stats={
            "patients": total_patients,
            "predictions": total_predictions,
            "high_risk": high_risk,
            "low_risk": low_risk,
        },
        activity=activity,
        recent_predictions=recent_predictions,
    )


@app.route("/admin/reports")
@role_required("admin")
def admin_reports():
    database = get_db()
    patients = database.execute(
        """
        SELECT p.patient_name, u.username, p.age, p.gender, p.created_at
        FROM patient_profiles AS p
        JOIN users AS u ON u.id = p.user_id
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    predictions = database.execute(
        """
        SELECT p.patient_name, r.prediction, r.probability, r.prediction_date
        FROM prediction_records AS r
        JOIN patient_profiles AS p ON p.id = r.patient_id
        ORDER BY r.prediction_date DESC, r.id DESC
        """
    ).fetchall()
    return render_template(
        "admin_reports.html", patients=patients, predictions=predictions
    )


def create_csv_response(filename, headers, rows):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/admin/export/patients.csv")
@role_required("admin")
def export_patients():
    patients = get_db().execute(
        """
        SELECT p.patient_name, u.username, p.age, p.gender, p.created_at
        FROM patient_profiles AS p
        JOIN users AS u ON u.id = p.user_id
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    return create_csv_response(
        "patient_records.csv",
        ["Patient Name", "Username", "Age", "Gender", "Registration Date"],
        [
            [row["patient_name"], row["username"], row["age"], row["gender"], row["created_at"]]
            for row in patients
        ],
    )


@app.route("/admin/export/predictions.csv")
@role_required("admin")
def export_predictions():
    predictions = get_db().execute(
        """
        SELECT p.patient_name, r.prediction, r.probability, r.prediction_date
        FROM prediction_records AS r
        JOIN patient_profiles AS p ON p.id = r.patient_id
        ORDER BY r.prediction_date DESC, r.id DESC
        """
    ).fetchall()
    return create_csv_response(
        "prediction_records.csv",
        ["Patient Name", "Prediction", "Probability", "Prediction Date"],
        [
            [
                row["patient_name"],
                "High Risk" if row["prediction"] else "Low Risk",
                row["probability"],
                row["prediction_date"],
            ]
            for row in predictions
        ],
    )


init_db()

if __name__ == "__main__":
    app.run(debug=True)
