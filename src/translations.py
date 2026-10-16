"""
src/translations.py
===================
English and Hindi interface translations.
All UI strings are stored here in a dictionary keyed by language code.
Usage:
    from src.translations import T, get_text
    text = get_text("predict_button", lang="hi")
"""

TRANSLATIONS = {
    "en": {
        # ── Navigation ──────────────────────────────────────────────────────────
        "nav_home":         "Home",
        "nav_predict":      "Predict Yield",
        "nav_batch":        "Batch Predictions",
        "nav_explorer":     "Dataset Explorer",
        "nav_performance":  "Model Performance",
        "nav_training":     "Model Training",
        "nav_history":      "Prediction History",
        "nav_about":        "About Project",

        # ── Common ──────────────────────────────────────────────────────────────
        "app_title":        "AI Precision Agriculture",
        "app_subtitle":     "Crop Yield Optimization System",
        "loading":          "Loading...",
        "error":            "Error",
        "warning":          "Warning",
        "success":          "Success",
        "no_data":          "No data available.",
        "select_option":    "Select an option",
        "submit":           "Submit",
        "reset":            "Reset",
        "download":         "Download",
        "clear":            "Clear",
        "confirm":          "Confirm",
        "cancel":           "Cancel",
        "language":         "Language",

        # ── Home ────────────────────────────────────────────────────────────────
        "home_title":           "AI-Powered Crop Yield Prediction",
        "home_subtitle":        "Precision Agriculture for Indian Farming",
        "home_description":     "This system analyses historical agricultural data across Indian states and trains regression models to estimate crop yield. It is a prediction prototype, not a validated farming-advisory system.",
        "home_model_status":    "Model Status",
        "home_model_ready":     "✅ Trained model available",
        "home_model_missing":   "⚠️ No trained model found — run train_model.py first",
        "home_dataset_stats":   "Dataset Overview",
        "home_total_records":   "Total Records",
        "home_crops":           "Crops",
        "home_states":          "States",
        "home_years":           "Years Covered",
        "home_disclaimer":      "⚠️ Disclaimer: Predictions are estimates based on historical patterns. They are not guaranteed outcomes and should not be used as the sole basis for farming decisions.",

        # ── Predict ─────────────────────────────────────────────────────────────
        "predict_title":        "Predict Crop Yield",
        "predict_subtitle":     "Enter crop details to get a yield estimate",
        "predict_crop":         "Crop",
        "predict_season":       "Season",
        "predict_state":        "State",
        "predict_year":         "Crop Year",
        "predict_area":         "Cultivated Area (hectares)",
        "predict_rainfall":     "Annual Rainfall (mm)",
        "predict_fertilizer":   "Fertilizer Used (kg)",
        "predict_pesticide":    "Pesticide Used (kg)",
        "predict_button":       "Predict Yield",
        "predict_result":       "Predicted Yield",
        "predict_unit":         "tonnes per hectare",
        "predict_model_used":   "Model used",
        "predict_disclaimer":   "This is an estimate based on historical data. Actual yield depends on many additional factors.",
        "predict_out_of_range": "⚠️ Some inputs are outside the training data range. Prediction may be less reliable.",
        "predict_listen":       "🔊 Listen to Result",
        "predict_stop":         "⏹ Stop",
        "predict_mic":          "🎤 Voice Input",
        "predict_mic_hint":     "Click to speak your input (supported in Chrome/Edge)",
        "predict_recognized":   "Recognized text",
        "predict_apply":        "Apply to Form",
        "predict_invalid_area": "Area must be greater than 0.",
        "predict_invalid_rainfall": "Rainfall must be between 0 and 10,000 mm.",
        "predict_invalid_year": "Please enter a valid year between 1990 and 2030.",
        "predict_select_crop":  "Please select a crop.",
        "predict_select_season":"Please select a season.",
        "predict_select_state": "Please select a state.",

        # ── Batch ────────────────────────────────────────────────────────────────
        "batch_title":          "Batch Predictions",
        "batch_subtitle":       "Upload a CSV to predict yield for multiple records",
        "batch_upload":         "Upload CSV File",
        "batch_upload_hint":    "Max 5 MB. Must contain: Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide",
        "batch_preview":        "Preview (first 5 rows)",
        "batch_valid_rows":     "Valid rows",
        "batch_invalid_rows":   "Invalid / rejected rows",
        "batch_run":            "Run Predictions",
        "batch_results":        "Prediction Results",
        "batch_download":       "Download Results CSV",
        "batch_summary":        "Prediction Summary",
        "batch_error_type":     "File must be a CSV.",
        "batch_error_size":     "File exceeds 5 MB limit.",
        "batch_error_cols":     "Missing required columns",
        "batch_no_valid":       "No valid rows to predict.",

        # ── Explorer ─────────────────────────────────────────────────────────────
        "explorer_title":       "Dataset Explorer",
        "explorer_subtitle":    "Explore the agricultural dataset",
        "explorer_shape":       "Dataset Shape",
        "explorer_rows":        "rows",
        "explorer_cols":        "columns",
        "explorer_filter":      "Filter Options",
        "explorer_filter_crop": "Filter by Crop",
        "explorer_filter_state":"Filter by State",
        "explorer_filter_season":"Filter by Season",
        "explorer_filter_year": "Filter by Year Range",
        "explorer_stats":       "Summary Statistics",
        "explorer_missing":     "Missing Value Analysis",
        "explorer_distribution":"Yield Distribution",
        "explorer_crop_compare":"Crop-wise Yield Comparison",
        "explorer_season_compare":"Season-wise Yield",
        "explorer_year_trend":  "Year-wise Trend",
        "explorer_state_compare":"State-wise Comparison",
        "explorer_correlation": "Correlation Heatmap",
        "explorer_note_corr":   "Note: Correlation measures statistical association, not causation.",

        # ── Performance ──────────────────────────────────────────────────────────
        "perf_title":           "Model Performance",
        "perf_subtitle":        "Comparison of trained regression models",
        "perf_mae":             "MAE (Mean Absolute Error)",
        "perf_rmse":            "RMSE (Root Mean Squared Error)",
        "perf_r2":              "R² (Coefficient of Determination)",
        "perf_mae_explain":     "Lower is better. Average magnitude of prediction errors.",
        "perf_rmse_explain":    "Lower is better. Penalises large errors more than MAE.",
        "perf_r2_explain":      "Higher is better (max 1.0). Proportion of variance explained. Can be negative for poor models.",
        "perf_best_model":      "Selected Model",
        "perf_test_size":       "Test set size",
        "perf_train_size":      "Training set size",
        "perf_actual_vs_pred":  "Actual vs. Predicted Yield",
        "perf_residuals":       "Residuals Plot",
        "perf_feat_importance": "Feature Importance",
        "perf_feat_note":       "Feature importance indicates which inputs the model relies on most. It does not prove that changing a feature will change yield.",
        "perf_limitation":      "Limitation: These metrics are from a single random split. Performance may vary with different splits or on new data.",

        # ── Training ─────────────────────────────────────────────────────────────
        "train_title":          "Model Training",
        "train_subtitle":       "Train and evaluate regression models",
        "train_status":         "Training Status",
        "train_last_run":       "Last Training Run",
        "train_button":         "Retrain Models (Local Only)",
        "train_warning":        "⚠️ Retraining is disabled in public deployment. Run train_model.py locally.",
        "train_running":        "Training in progress...",
        "train_success":        "Training completed successfully.",
        "train_failed":         "Training failed. See error details.",

        # ── History ──────────────────────────────────────────────────────────────
        "history_title":        "Prediction History",
        "history_subtitle":     "Recent predictions from this session",
        "history_empty":        "No predictions yet in this session.",
        "history_clear":        "Clear History",
        "history_cleared":      "History cleared.",
        "history_timestamp":    "Timestamp",
        "history_crop":         "Crop",
        "history_yield":        "Predicted Yield",
        "history_model":        "Model",

        # ── About ────────────────────────────────────────────────────────────────
        "about_title":          "About This Project",
        "about_problem":        "Problem Statement",
        "about_objectives":     "Objectives",
        "about_dataset":        "Dataset",
        "about_algorithms":     "Algorithms",
        "about_metrics":        "Evaluation Metrics",
        "about_workflow":       "System Workflow",
        "about_scope":          "Current Scope & Limitations",
        "about_future":         "Future Extensions",

        # ── Voice ────────────────────────────────────────────────────────────────
        "voice_lang_en":        "English",
        "voice_lang_hi":        "Hindi",
        "voice_permission":     "Microphone permission is required for voice input.",
        "voice_not_supported":  "Speech recognition is not supported in this browser. Try Chrome or Edge.",
        "voice_tts_not_supported": "Speech synthesis is not available in this browser.",
        "voice_no_hindi":       "Hindi voice may not be available on this device. English will be used.",
        "voice_privacy":        "Note: Browser speech recognition may process audio on external servers.",
        "voice_recording":      "🔴 Recording... Speak now.",
        "voice_processing":     "Processing speech...",
        "voice_confirm_msg":    "Apply recognized text to form?",
    },

    "hi": {
        # ── Navigation ──────────────────────────────────────────────────────────
        "nav_home":         "होम",
        "nav_predict":      "उपज अनुमान",
        "nav_batch":        "बैच अनुमान",
        "nav_explorer":     "डेटासेट एक्सप्लोरर",
        "nav_performance":  "मॉडल प्रदर्शन",
        "nav_training":     "मॉडल प्रशिक्षण",
        "nav_history":      "अनुमान इतिहास",
        "nav_about":        "परियोजना के बारे में",

        # ── Common ──────────────────────────────────────────────────────────────
        "app_title":        "AI परिशुद्ध कृषि",
        "app_subtitle":     "फसल उपज अनुकूलन प्रणाली",
        "loading":          "लोड हो रहा है...",
        "error":            "त्रुटि",
        "warning":          "चेतावनी",
        "success":          "सफलता",
        "no_data":          "कोई डेटा उपलब्ध नहीं।",
        "select_option":    "विकल्प चुनें",
        "submit":           "सबमिट करें",
        "reset":            "रीसेट करें",
        "download":         "डाउनलोड करें",
        "clear":            "साफ़ करें",
        "confirm":          "पुष्टि करें",
        "cancel":           "रद्द करें",
        "language":         "भाषा",

        # ── Home ────────────────────────────────────────────────────────────────
        "home_title":           "AI-संचालित फसल उपज अनुमान",
        "home_subtitle":        "भारतीय खेती के लिए परिशुद्ध कृषि",
        "home_description":     "यह प्रणाली भारतीय राज्यों के ऐतिहासिक कृषि डेटा का विश्लेषण करती है और फसल उपज का अनुमान लगाने के लिए रिग्रेशन मॉडल प्रशिक्षित करती है। यह एक पूर्वानुमान प्रोटोटाइप है, सत्यापित कृषि सलाह प्रणाली नहीं।",
        "home_model_status":    "मॉडल स्थिति",
        "home_model_ready":     "✅ प्रशिक्षित मॉडल उपलब्ध",
        "home_model_missing":   "⚠️ कोई प्रशिक्षित मॉडल नहीं मिला — पहले train_model.py चलाएं",
        "home_dataset_stats":   "डेटासेट अवलोकन",
        "home_total_records":   "कुल रिकॉर्ड",
        "home_crops":           "फसलें",
        "home_states":          "राज्य",
        "home_years":           "वर्ष",
        "home_disclaimer":      "⚠️ अस्वीकरण: अनुमान ऐतिहासिक पैटर्न पर आधारित हैं। ये गारंटीकृत परिणाम नहीं हैं।",

        # ── Predict ─────────────────────────────────────────────────────────────
        "predict_title":        "फसल उपज का अनुमान लगाएं",
        "predict_subtitle":     "उपज अनुमान के लिए फसल विवरण दर्ज करें",
        "predict_crop":         "फसल",
        "predict_season":       "मौसम",
        "predict_state":        "राज्य",
        "predict_year":         "फसल वर्ष",
        "predict_area":         "खेती क्षेत्र (हेक्टेयर)",
        "predict_rainfall":     "वार्षिक वर्षा (मिमी)",
        "predict_fertilizer":   "उर्वरक (किग्रा)",
        "predict_pesticide":    "कीटनाशक (किग्रा)",
        "predict_button":       "उपज का अनुमान लगाएं",
        "predict_result":       "अनुमानित उपज",
        "predict_unit":         "टन प्रति हेक्टेयर",
        "predict_model_used":   "उपयोग किया गया मॉडल",
        "predict_disclaimer":   "यह ऐतिहासिक डेटा पर आधारित अनुमान है। वास्तविक उपज कई अतिरिक्त कारकों पर निर्भर करती है।",
        "predict_out_of_range": "⚠️ कुछ इनपुट प्रशिक्षण डेटा की सीमा से बाहर हैं। अनुमान कम विश्वसनीय हो सकता है।",
        "predict_listen":       "🔊 परिणाम सुनें",
        "predict_stop":         "⏹ रोकें",
        "predict_mic":          "🎤 वॉइस इनपुट",
        "predict_mic_hint":     "बोलने के लिए क्लिक करें (Chrome/Edge में समर्थित)",
        "predict_recognized":   "पहचाना गया पाठ",
        "predict_apply":        "फॉर्म में लागू करें",
        "predict_invalid_area": "क्षेत्र 0 से अधिक होना चाहिए।",
        "predict_invalid_rainfall": "वर्षा 0 से 10,000 मिमी के बीच होनी चाहिए।",
        "predict_invalid_year": "1990 और 2030 के बीच वैध वर्ष दर्ज करें।",
        "predict_select_crop":  "कृपया एक फसल चुनें।",
        "predict_select_season":"कृपया एक मौसम चुनें।",
        "predict_select_state": "कृपया एक राज्य चुनें।",

        # ── Batch ────────────────────────────────────────────────────────────────
        "batch_title":          "बैच अनुमान",
        "batch_subtitle":       "एकाधिक रिकॉर्ड के लिए CSV अपलोड करें",
        "batch_upload":         "CSV फ़ाइल अपलोड करें",
        "batch_upload_hint":    "अधिकतम 5 MB। आवश्यक कॉलम: Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide",
        "batch_preview":        "पूर्वावलोकन (पहली 5 पंक्तियाँ)",
        "batch_valid_rows":     "वैध पंक्तियाँ",
        "batch_invalid_rows":   "अमान्य पंक्तियाँ",
        "batch_run":            "अनुमान चलाएं",
        "batch_results":        "अनुमान परिणाम",
        "batch_download":       "परिणाम CSV डाउनलोड करें",
        "batch_summary":        "अनुमान सारांश",
        "batch_error_type":     "फ़ाइल CSV होनी चाहिए।",
        "batch_error_size":     "फ़ाइल 5 MB की सीमा से अधिक है।",
        "batch_error_cols":     "आवश्यक कॉलम गायब हैं",
        "batch_no_valid":       "अनुमान के लिए कोई वैध पंक्ति नहीं।",

        # ── Explorer ─────────────────────────────────────────────────────────────
        "explorer_title":       "डेटासेट एक्सप्लोरर",
        "explorer_subtitle":    "कृषि डेटासेट का अन्वेषण करें",
        "explorer_shape":       "डेटासेट आकार",
        "explorer_rows":        "पंक्तियाँ",
        "explorer_cols":        "कॉलम",
        "explorer_filter":      "फ़िल्टर विकल्प",
        "explorer_filter_crop": "फसल द्वारा फ़िल्टर",
        "explorer_filter_state":"राज्य द्वारा फ़िल्टर",
        "explorer_filter_season":"मौसम द्वारा फ़िल्टर",
        "explorer_filter_year": "वर्ष सीमा द्वारा फ़िल्टर",
        "explorer_stats":       "सारांश आँकड़े",
        "explorer_missing":     "अनुपस्थित मान विश्लेषण",
        "explorer_distribution":"उपज वितरण",
        "explorer_crop_compare":"फसल-वार उपज तुलना",
        "explorer_season_compare":"मौसम-वार उपज",
        "explorer_year_trend":  "वर्ष-वार प्रवृत्ति",
        "explorer_state_compare":"राज्य-वार तुलना",
        "explorer_correlation": "सहसंबंध हीटमैप",
        "explorer_note_corr":   "नोट: सहसंबंध सांख्यिकीय संबंध को मापता है, कारण-प्रभाव को नहीं।",

        # ── Performance ──────────────────────────────────────────────────────────
        "perf_title":           "मॉडल प्रदर्शन",
        "perf_subtitle":        "प्रशिक्षित रिग्रेशन मॉडल की तुलना",
        "perf_mae":             "MAE (मध्य पूर्ण त्रुटि)",
        "perf_rmse":            "RMSE (वर्गमूल माध्य वर्ग त्रुटि)",
        "perf_r2":              "R² (निर्धारण गुणांक)",
        "perf_mae_explain":     "कम बेहतर है। पूर्वानुमान त्रुटियों की औसत मात्रा।",
        "perf_rmse_explain":    "कम बेहतर है। बड़ी त्रुटियों को MAE से अधिक दंडित करता है।",
        "perf_r2_explain":      "अधिक बेहतर है (अधिकतम 1.0)। खराब मॉडल के लिए नकारात्मक हो सकता है।",
        "perf_best_model":      "चयनित मॉडल",
        "perf_test_size":       "परीक्षण सेट आकार",
        "perf_train_size":      "प्रशिक्षण सेट आकार",
        "perf_actual_vs_pred":  "वास्तविक बनाम अनुमानित उपज",
        "perf_residuals":       "अवशेष भूखंड",
        "perf_feat_importance": "विशेषता महत्व",
        "perf_feat_note":       "विशेषता महत्व दर्शाता है कि मॉडल किन इनपुट पर सबसे अधिक निर्भर करता है। यह साबित नहीं करता कि किसी विशेषता को बदलने से उपज बदलेगी।",
        "perf_limitation":      "सीमा: ये मेट्रिक्स एकल यादृच्छिक विभाजन से हैं। विभिन्न विभाजन के साथ प्रदर्शन भिन्न हो सकता है।",

        # ── Training ─────────────────────────────────────────────────────────────
        "train_title":          "मॉडल प्रशिक्षण",
        "train_subtitle":       "रिग्रेशन मॉडल प्रशिक्षित और मूल्यांकन करें",
        "train_status":         "प्रशिक्षण स्थिति",
        "train_last_run":       "अंतिम प्रशिक्षण",
        "train_button":         "मॉडल पुनः प्रशिक्षित करें (केवल स्थानीय)",
        "train_warning":        "⚠️ सार्वजनिक तैनाती में पुनः प्रशिक्षण अक्षम है। train_model.py स्थानीय रूप से चलाएं।",
        "train_running":        "प्रशिक्षण प्रगति में है...",
        "train_success":        "प्रशिक्षण सफलतापूर्वक पूर्ण हुआ।",
        "train_failed":         "प्रशिक्षण विफल हुआ।",

        # ── History ──────────────────────────────────────────────────────────────
        "history_title":        "अनुमान इतिहास",
        "history_subtitle":     "इस सत्र के हाल के अनुमान",
        "history_empty":        "इस सत्र में अभी तक कोई अनुमान नहीं।",
        "history_clear":        "इतिहास साफ़ करें",
        "history_cleared":      "इतिहास साफ़ किया गया।",
        "history_timestamp":    "समय",
        "history_crop":         "फसल",
        "history_yield":        "अनुमानित उपज",
        "history_model":        "मॉडल",

        # ── About ────────────────────────────────────────────────────────────────
        "about_title":          "परियोजना के बारे में",
        "about_problem":        "समस्या विवरण",
        "about_objectives":     "उद्देश्य",
        "about_dataset":        "डेटासेट",
        "about_algorithms":     "एल्गोरिदम",
        "about_metrics":        "मूल्यांकन मेट्रिक्स",
        "about_workflow":       "सिस्टम वर्कफ़्लो",
        "about_scope":          "वर्तमान दायरा और सीमाएं",
        "about_future":         "भविष्य के विस्तार",

        # ── Voice ────────────────────────────────────────────────────────────────
        "voice_lang_en":        "English",
        "voice_lang_hi":        "हिन्दी",
        "voice_permission":     "वॉइस इनपुट के लिए माइक्रोफ़ोन अनुमति आवश्यक है।",
        "voice_not_supported":  "इस ब्राउज़र में वाक् पहचान समर्थित नहीं है। Chrome या Edge आज़माएं।",
        "voice_tts_not_supported": "इस ब्राउज़र में वाक् संश्लेषण उपलब्ध नहीं है।",
        "voice_no_hindi":       "इस डिवाइस पर हिंदी आवाज़ उपलब्ध नहीं हो सकती। अंग्रेज़ी का उपयोग किया जाएगा।",
        "voice_privacy":        "नोट: ब्राउज़र वाक् पहचान बाहरी सर्वर पर ऑडियो संसाधित कर सकती है।",
        "voice_recording":      "🔴 रिकॉर्डिंग... अभी बोलें।",
        "voice_processing":     "वाक् संसाधित हो रहा है...",
        "voice_confirm_msg":    "पहचाना गया पाठ फॉर्म में लागू करें?",
    },
}

# ── helper function ───────────────────────────────────────────────────────────

def get_text(key: str, lang: str = "en") -> str:
    """Return translated text for key in the given language.
    Falls back to English if the key is missing in the target language.
    Returns the key itself if not found in English either.
    """
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    text = lang_dict.get(key)
    if text is None:
        # fallback to English
        text = TRANSLATIONS["en"].get(key)
    if text is None:
        # last resort: return the key so broken keys are visible
        return f"[{key}]"
    return text


def T(key: str, lang: str = "en") -> str:
    """Shorthand alias for get_text."""
    return get_text(key, lang)


# ── available languages ───────────────────────────────────────────────────────
LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
}

LANGUAGE_CODES = {
    "en": "en-IN",   # BCP-47 tag for browser speech APIs
    "hi": "hi-IN",
}
