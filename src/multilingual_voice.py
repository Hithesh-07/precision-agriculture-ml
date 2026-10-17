"""
src/multilingual_voice.py
=========================
Multilingual Speech Recognition & Translation Engine for CropZen.
Supports English, Telugu (తెలుగు), and Hindi (हिन्दी).

Key Capabilities:
1. Speech Language Recognition: Supports te-IN (Telugu), hi-IN (Hindi), en-IN (English).
2. Entity Extraction: Parses spoken Indian languages to extract Crop, State, Season, Area,
   Rainfall, Fertilizer, and Year directly into standard English model parameters.
3. Multilingual Answer Generation: Formats predictions and advisory into natural Telugu,
   Hindi, or English speech for browser Text-to-Speech (TTS).
4. Numeral Normalization: Converts Telugu numerals (౦-౯) and Devanagari numerals (०-९)
   to standard Hindu-Arabic digits (0-9).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


# ── Crop Multilingual Mappings ────────────────────────────────────────────────

CROP_DISPLAY_MAP: Dict[str, Dict[str, str]] = {
    "Rice": {"en": "Rice", "te": "వరి", "hi": "चावल / धान"},
    "Wheat": {"en": "Wheat", "te": "గోధుమ", "hi": "गेहूं"},
    "Maize": {"en": "Maize", "te": "మొక్కజొన్న", "hi": "मक्का"},
    "Cotton(lint)": {"en": "Cotton", "te": "పత్తి", "hi": "कपास"},
    "Sugarcane": {"en": "Sugarcane", "te": "చెరకు", "hi": "गन्ना"},
    "Groundnut": {"en": "Groundnut", "te": "వేరుశనగ", "hi": "मूंगफली"},
    "Jowar": {"en": "Jowar", "te": "జొన్న", "hi": "ज्वार"},
    "Bajra": {"en": "Bajra", "te": "సజ్జలు / బాజ్రా", "hi": "बाजरा"},
    "Ragi": {"en": "Ragi", "te": "రాగి / తైదలు", "hi": "रागी / मडुआ"},
    "Banana": {"en": "Banana", "te": "అరటి", "hi": "केला"},
    "Dry chillies": {"en": "Dry chillies", "te": "ఎండుమిర్చి", "hi": "सूखी लाल मिर्च"},
    "Onion": {"en": "Onion", "te": "ఉల్లిపాయ", "hi": "प्याज"},
    "Potato": {"en": "Potato", "te": "బంగాళాదుంప", "hi": "आलू"},
    "Turmeric": {"en": "Turmeric", "te": "పసుపు", "hi": "हल्दी"},
    "Sunflower": {"en": "Sunflower", "te": "పొద్దుతిరుగుడు", "hi": "सूरजमुखी"},
    "Soyabean": {"en": "Soyabean", "te": "సోయాబీన్", "hi": "सोयाबीन"},
    "Tobacco": {"en": "Tobacco", "te": "పొగాకు", "hi": "तंबाकू"},
    "Ginger": {"en": "Ginger", "te": "అల్లం", "hi": "अदरक"},
    "Garlic": {"en": "Garlic", "te": "వెల్లుల్లి", "hi": "लहसुन"},
    "Gram": {"en": "Gram (Chickpea)", "te": "శనగలు", "hi": "चना"},
    "Arhar/Tur": {"en": "Arhar/Tur (Pigeon Pea)", "te": "కందులు", "hi": "अरहर / तुअर"},
    "Moong(Green Gram)": {"en": "Moong (Green Gram)", "te": "పెసలు", "hi": "मूंग"},
    "Urad": {"en": "Urad (Black Gram)", "te": "మినుములు", "hi": "उड़द"},
    "Coconut": {"en": "Coconut", "te": "కొబ్బరి", "hi": "नारियल"},
    "Cashewnut": {"en": "Cashewnut", "te": "జీడిపప్పు", "hi": "काजू"},
    "Cardamom": {"en": "Cardamom", "te": "యాలకులు", "hi": "इलायची"},
    "Black pepper": {"en": "Black pepper", "te": "మిరియాలు", "hi": "काली मिर्च"},
    "Coriander": {"en": "Coriander", "te": "ధనియాలు", "hi": "धनिया"},
    "Barley": {"en": "Barley", "te": "బార్లీ / యవలు", "hi": "जौ"},
    "Castor seed": {"en": "Castor seed", "te": "ఆముదం", "hi": "अरंडी"},
    "Rapeseed &Mustard": {"en": "Mustard", "te": "ఆవాలు", "hi": "सरसों / राई"},
    "Jute": {"en": "Jute", "te": "జనపనార", "hi": "पटसन / जूट"},
    "Arecanut": {"en": "Arecanut", "te": "పోక చెక్క", "hi": "सुपारी"},
    "Sesamum": {"en": "Sesame", "te": "నువ్వులు", "hi": "तिल"},
    "Sweet potato": {"en": "Sweet potato", "te": "చిలగడదుంప", "hi": "शकरकंद"},
    "Tapioca": {"en": "Tapioca", "te": "కర్రపెండలం", "hi": "कसावा / टैपिओका"},
    "Horse-gram": {"en": "Horse-gram", "te": "ఉలవలు", "hi": "कुलथी"},
    "Masoor": {"en": "Masoor (Red Lentil)", "te": "మసూర్ పప్పు", "hi": "मसूर"},
    "Linseed": {"en": "Linseed / Flax", "te": "అవిసె గింజలు", "hi": "अलसी"},
    "Safflower": {"en": "Safflower", "te": "కుసుమలు", "hi": "कुसुम"},
    "Small millets": {"en": "Small millets", "te": "చిరుధాన్యాలు", "hi": "लघु बाजरा"},
    "Cowpea(Lobia)": {"en": "Cowpea", "te": "బొబ్బర్లు / అలసందలు", "hi": "लोबिया"},
}

# Reverse lookup table: Any synonym/translation in Telugu, Hindi, or English -> Standard Model Name
CROP_SYNONYMS: Dict[str, str] = {
    # Telugu
    "వరి": "Rice", "వరి పంట": "Rice", "బియ్యం": "Rice", "vari": "Rice",
    "గోధుమ": "Wheat", "గోధుమలు": "Wheat", "godhuma": "Wheat",
    "మొక్కజొన్న": "Maize", "మొక్క జొన్న": "Maize", "mokkajonna": "Maize",
    "పత్తి": "Cotton(lint)", "పత్తి పంట": "Cotton(lint)", "patti": "Cotton(lint)",
    "చెరకు": "Sugarcane", "చెఱకు": "Sugarcane", "cheraku": "Sugarcane",
    "వేరుశనగ": "Groundnut", "వేరు శనగ": "Groundnut", "పల్లీ": "Groundnut", "verusanaga": "Groundnut",
    "జొన్న": "Jowar", "జొన్నలు": "Jowar", "jonna": "Jowar",
    "సజ్జలు": "Bajra", "సజ్జ": "Bajra", "బాజ్రా": "Bajra", "sajjalu": "Bajra",
    "రాగి": "Ragi", "రాగులు": "Ragi", "తైదలు": "Ragi", "ragi": "Ragi",
    "అరటి": "Banana", "అరటి తోట": "Banana", "arati": "Banana",
    "మిరప": "Dry chillies", "ఎండుమిర్చి": "Dry chillies", "మిర్చి": "Dry chillies", "mirapa": "Dry chillies",
    "ఉల్లిపాయ": "Onion", "ఉల్లి": "Onion", "ullipaya": "Onion",
    "బంగాళాదుంప": "Potato", "ఆలూ": "Potato", "bangaladumpa": "Potato",
    "పసుపు": "Turmeric", "pasupu": "Turmeric",
    "పొద్దుతిరుగుడు": "Sunflower", "పొద్దు తిరుగుడు": "Sunflower", "podduthirugudu": "Sunflower",
    "సోయాబీన్": "Soyabean", "సోయా": "Soyabean", "soyabean": "Soyabean",
    "పొగాకు": "Tobacco", "pogaku": "Tobacco",
    "అల్లం": "Ginger", "allam": "Ginger",
    "వెల్లుల్లి": "Garlic", "వెల్లుల్లిపాయ": "Garlic", "vellulli": "Garlic",
    "శనగలు": "Gram", "శనగ": "Gram", "senagalu": "Gram",
    "కందులు": "Arhar/Tur", "కంది": "Arhar/Tur", "తువర్": "Arhar/Tur", "kandulu": "Arhar/Tur",
    "పెసలు": "Moong(Green Gram)", "పెసర": "Moong(Green Gram)", "pesalu": "Moong(Green Gram)",
    "మినుములు": "Urad", "మినుము": "Urad", "ఉద్ది": "Urad", "minumulu": "Urad",
    "కొబ్బరి": "Coconut", "కొబ్బరికాయ": "Coconut", "kobbari": "Coconut",
    "జీడిపప్పు": "Cashewnut", "జీడిమామిడి": "Cashewnut",
    "యాలకులు": "Cardamom", "ఏలకులు": "Cardamom",
    "మిరియాలు": "Black pepper", "నల్ల మిరియాలు": "Black pepper",
    "ధనియాలు": "Coriander", "కొత్తిమీర": "Coriander",
    "బార్లీ": "Barley", "యవలు": "Barley",
    "ఆముదం": "Castor seed", "ఆముదాలు": "Castor seed",
    "ఆవాలు": "Rapeseed &Mustard",
    "జనపనార": "Jute", "జనము": "Jute",
    "పోక చెక్క": "Arecanut", "వక్క": "Arecanut",
    "నువ్వులు": "Sesamum",
    "చిలగడదుంప": "Sweet potato", "గెనుసు": "Sweet potato",
    "ఉలవలు": "Horse-gram",
    "మసూర్": "Masoor",
    "అవిసెలు": "Linseed",

    # Hindi
    "चावल": "Rice", "धान": "Rice", "chawal": "Rice", "dhan": "Rice",
    "गेहूं": "Wheat", "गेंहू": "Wheat", "gehun": "Wheat", "gehu": "Wheat",
    "मक्का": "Maize", "मकई": "Maize", "makka": "Maize",
    "कपास": "Cotton(lint)", "रुई": "Cotton(lint)", "kapas": "Cotton(lint)",
    "गन्ना": "Sugarcane", "ईख": "Sugarcane", "ganna": "Sugarcane",
    "मूंगफली": "Groundnut", "mungfali": "Groundnut",
    "ज्वार": "Jowar", "jowar": "Jowar",
    "बाजरा": "Bajra", "bajra": "Bajra",
    "रागी": "Ragi", "मडुआ": "Ragi",
    "केला": "Banana", "kela": "Banana",
    "मिर्च": "Dry chillies", "लाल मिर्च": "Dry chillies", "सूखी मिर्च": "Dry chillies",
    "प्याज": "Onion", "कांदा": "Onion", "pyaj": "Onion",
    "आलू": "Potato", "aaloo": "Potato", "alu": "Potato",
    "हल्दी": "Turmeric", "haldi": "Turmeric",
    "सूरजमुखी": "Sunflower", "surajmukhi": "Sunflower",
    "सोयाबीन": "Soyabean", "soyabean": "Soyabean",
    "तंबाकू": "Tobacco", "tambaku": "Tobacco",
    "अदरक": "Ginger", "adrak": "Ginger",
    "लहसुन": "Garlic", "lahsun": "Garlic",
    "चना": "Gram", "चने": "Gram", "chana": "Gram",
    "अरहर": "Arhar/Tur", "तुअर": "Arhar/Tur", "तुवर": "Arhar/Tur", "arhar": "Arhar/Tur",
    "मूंग": "Moong(Green Gram)", "mung": "Moong(Green Gram)",
    "उड़द": "Urad", "urad": "Urad",
    "नारियल": "Coconut", "nariyal": "Coconut",
    "काजू": "Cashewnut", "kaju": "Cashewnut",
    "इलायची": "Cardamom", "elachi": "Cardamom",
    "काली मिर्च": "Black pepper", "kali mirch": "Black pepper",
    "धनिया": "Coriander", "dhaniya": "Coriander",
    "जौ": "Barley", "jau": "Barley",
    "अरंडी": "Castor seed", "arandi": "Castor seed",
    "सरसों": "Rapeseed &Mustard", "राई": "Rapeseed &Mustard", "sarson": "Rapeseed &Mustard",
    "पटसन": "Jute", "जूट": "Jute", "jute": "Jute",
    "सुपारी": "Arecanut", "supari": "Arecanut",
    "तिल": "Sesamum", "til": "Sesamum",
    "शकरकंद": "Sweet potato", "shakarkand": "Sweet potato",
    "मसूर": "Masoor", "masoor": "Masoor",
    "अलसी": "Linseed", "alsi": "Linseed",

    # English standard and aliases
    "rice": "Rice", "paddy": "Rice",
    "wheat": "Wheat",
    "maize": "Maize", "corn": "Maize",
    "cotton": "Cotton(lint)",
    "sugarcane": "Sugarcane",
    "groundnut": "Groundnut", "peanut": "Groundnut",
    "dry chillies": "Dry chillies", "chillies": "Dry chillies", "chilli": "Dry chillies",
    "onion": "Onion",
    "potato": "Potato",
    "turmeric": "Turmeric",
    "sunflower": "Sunflower",
    "banana": "Banana",
    "garlic": "Garlic",
    "ginger": "Ginger",
    "gram": "Gram", "chickpea": "Gram",
    "moong": "Moong(Green Gram)",
    "soyabean": "Soyabean", "soybean": "Soyabean",
}


# ── State Multilingual Mappings ───────────────────────────────────────────────

STATE_DISPLAY_MAP: Dict[str, Dict[str, str]] = {
    "Andhra Pradesh": {"en": "Andhra Pradesh", "te": "ఆంధ్రప్రదేశ్", "hi": "आंध्र प्रदेश"},
    "Telangana": {"en": "Telangana", "te": "తెలంగాణ", "hi": "तेलंगाना"},
    "Punjab": {"en": "Punjab", "te": "పంజాబ్", "hi": "पंजाब"},
    "Haryana": {"en": "Haryana", "te": "హర్యానా", "hi": "हरियाणा"},
    "Karnataka": {"en": "Karnataka", "te": "కర్ణాటక", "hi": "कर्नाटक"},
    "Tamil Nadu": {"en": "Tamil Nadu", "te": "తమిళనాడు", "hi": "तमिलनाडु"},
    "Maharashtra": {"en": "Maharashtra", "te": "మహారాష్ట్ర", "hi": "महाराष्ट्र"},
    "Gujarat": {"en": "Gujarat", "te": "గుజరాత్", "hi": "गुजरात"},
    "Uttar Pradesh": {"en": "Uttar Pradesh", "te": "ఉత్తర ప్రదేశ్", "hi": "उत्तर प्रदेश"},
    "Madhya Pradesh": {"en": "Madhya Pradesh", "te": "మధ్యప్రదేశ్", "hi": "मध्य प्रदेश"},
    "Bihar": {"en": "Bihar", "te": "బీహార్", "hi": "बिहार"},
    "West Bengal": {"en": "West Bengal", "te": "పశ్చిమ బెంగాల్", "hi": "पश्चिम बंगाल"},
    "Odisha": {"en": "Odisha", "te": "ఒడిశా", "hi": "ओडिशा"},
    "Kerala": {"en": "Kerala", "te": "కేరళ", "hi": "केरल"},
    "Rajasthan": {"en": "Rajasthan", "te": "రాజస్థాన్", "hi": "राजस्थान"},
    "Assam": {"en": "Assam", "te": "అస్సాం", "hi": "असम"},
}

STATE_SYNONYMS: Dict[str, str] = {
    # Telugu
    "ఆంధ్రప్రదేశ్": "Andhra Pradesh", "ఆంధ్ర ప్రదేశ్": "Andhra Pradesh", "ఆంధ్ర": "Andhra Pradesh", "andhra": "Andhra Pradesh",
    "తెలంగాణ": "Telangana", "తెలంగాణా": "Telangana", "telangana": "Telangana",
    "పంజాబ్": "Punjab", "punjab": "Punjab",
    "హర్యానా": "Haryana", "haryana": "Haryana",
    "కర్ణాటక": "Karnataka", "కర్నాటక": "Karnataka", "karnataka": "Karnataka",
    "తమిళనాడు": "Tamil Nadu", "తమిళ నాడు": "Tamil Nadu", "tamil nadu": "Tamil Nadu",
    "మహారాష్ట్ర": "Maharashtra", "మహారాష్ట్ర": "Maharashtra", "maharashtra": "Maharashtra",
    "గుజరాత్": "Gujarat", "gujarat": "Gujarat",
    "ఉత్తర ప్రదేశ్": "Uttar Pradesh", "ఉత్తరప్రదేశ్": "Uttar Pradesh", "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "మధ్యప్రదేశ్": "Madhya Pradesh", "మధ్య ప్రదేశ్": "Madhya Pradesh", "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh",
    "బీహార్": "Bihar", "bihar": "Bihar",
    "పశ్చిమ బెంగాల్": "West Bengal", "బెంగాల్": "West Bengal", "west bengal": "West Bengal",
    "ఒడిశా": "Odisha", "ఒరిస్సా": "Odisha", "odisha": "Odisha",
    "కేరళ": "Kerala", "kerala": "Kerala",
    "అస్సాం": "Assam", "assam": "Assam",
    "ఛత్తీస్‌గఢ్": "Chhattisgarh", "ఛత్తీస్‌ఘర్": "Chhattisgarh",

    # Hindi
    "आंध्र प्रदेश": "Andhra Pradesh", "आन्ध्र प्रदेश": "Andhra Pradesh", "आंध्र": "Andhra Pradesh",
    "तेलंगाना": "Telangana",
    "पंजाब": "Punjab",
    "हरियाणा": "Haryana",
    "कर्नाटक": "Karnataka",
    "तमिलनाडु": "Tamil Nadu", "तमिल नाडु": "Tamil Nadu",
    "महाराष्ट्र": "Maharashtra",
    "गुजरात": "Gujarat",
    "उत्तर प्रदेश": "Uttar Pradesh", "यूपी": "Uttar Pradesh",
    "मध्य प्रदेश": "Madhya Pradesh", "एमपी": "Madhya Pradesh",
    "बिहार": "Bihar",
    "पश्चिम बंगाल": "West Bengal", "बंगाल": "West Bengal",
    "ओडिशा": "Odisha", "उड़ीसा": "Odisha",
    "केरल": "Kerala",
    "असम": "Assam",
    "छत्तीसगढ़": "Chhattisgarh",
}


# ── Season Multilingual Mappings ──────────────────────────────────────────────

SEASON_DISPLAY_MAP: Dict[str, Dict[str, str]] = {
    "Kharif": {"en": "Kharif", "te": "ఖరీఫ్ (వానకాలం)", "hi": "खरीफ"},
    "Rabi": {"en": "Rabi", "te": "రబీ (యాసంగి)", "hi": "रबी"},
    "Summer": {"en": "Summer", "te": "వేసవి", "hi": "ज़ायद / ग्रीष्म"},
    "Whole Year": {"en": "Whole Year", "te": "సంవత్సరం మొత్తం", "hi": "पूरे साल"},
    "Winter": {"en": "Winter", "te": "శీతాకాలం", "hi": "सर्दियों का मौसम"},
    "Autumn": {"en": "Autumn", "te": "శరదృతువు", "hi": "पतझड़"},
}

SEASON_SYNONYMS: Dict[str, str] = {
    # Telugu
    "ఖరీఫ్": "Kharif", "వానకాలం": "Kharif", "kharif": "Kharif",
    "రబీ": "Rabi", "యాసంగి": "Rabi", "శీతాకాల పంట": "Rabi", "rabi": "Rabi",
    "వేసవి": "Summer", "ఎండాకాలం": "Summer", "summer": "Summer",
    "సంవత్సరం మొత్తం": "Whole Year", "ఏడాదంతా": "Whole Year",
    "శీతాకాలం": "Winter", "winter": "Winter",
    "శరదృతువు": "Autumn", "autumn": "Autumn",

    # Hindi
    "खरीफ": "Kharif", "मानसून": "Kharif",
    "रबी": "Rabi", "सर्दी की फसल": "Rabi",
    "ज़ायद": "Summer", "जायद": "Summer", "गर्मी": "Summer", "ग्रीष्म": "Summer",
    "पूरे साल": "Whole Year", "सालाना": "Whole Year",
    "सर्दी": "Winter", "शीतकाल": "Winter",
    "पतझड़": "Autumn",
}


# ── Language Detection & Numeral Normalization ────────────────────────────────

def detect_language(text: str) -> str:
    """Detect whether input text is Telugu ('te'), Hindi ('hi'), or English ('en')."""
    if not text:
        return "en"
    # Telugu Unicode Block: U+0C00 to U+0C7F
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te"
    # Devanagari (Hindi) Unicode Block: U+0900 to U+097F
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def normalize_indian_numerals(text: str) -> str:
    """Convert Telugu numerals (౦-౯) and Hindi numerals (०-९) to ASCII 0-9."""
    # Telugu digits
    telugu_digits = "౦౧౨౩౪౫౬౭౮౯"
    # Devanagari digits
    hindi_digits = "०१२३४५६७८९"

    result = []
    for char in text:
        if char in telugu_digits:
            result.append(str(telugu_digits.index(char)))
        elif char in hindi_digits:
            result.append(str(hindi_digits.index(char)))
        else:
            result.append(char)
    return "".join(result)


# ── Entity Extraction Engine ──────────────────────────────────────────────────

def parse_multilingual_voice_text(text: str, default_lang: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses spoken voice transcript in Telugu, Hindi, or English.
    Extracts entities and maps them into model parameters:
    - Crop (standard English name)
    - State (standard English state)
    - Season (standard English season)
    - Area (hectares)
    - Annual_Rainfall (mm)
    - Crop_Year (integer)
    - Fertilizer (kg/ha or total)
    - Pesticide (kg/ha or total)
    """
    clean_text = normalize_indian_numerals(text or "").strip()
    detected_lang = detect_language(clean_text) if clean_text else (default_lang or "en")
    lower_text = clean_text.lower()

    extracted: Dict[str, Any] = {
        "detected_lang": detected_lang,
        "raw_text": text,
        "clean_text": clean_text,
        "crop": None,
        "crop_display": None,
        "state": None,
        "state_display": None,
        "season": None,
        "season_display": None,
        "year": None,
        "area": None,
        "rainfall": None,
        "fertilizer": None,
        "pesticide": None,
        "entities_found": [],
    }

    if not clean_text:
        return extracted

    # 1. Match Crop
    # Sort synonyms by length descending to match longest phrases first (e.g., 'dry chillies' before 'chillies')
    sorted_crop_syns = sorted(CROP_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)
    for syn, std_crop in sorted_crop_syns:
        pattern = r"\b" + re.escape(syn) + r"\b" if syn.isascii() else re.escape(syn)
        if re.search(pattern, clean_text, flags=re.IGNORECASE):
            extracted["crop"] = std_crop
            display_name = CROP_DISPLAY_MAP.get(std_crop, {}).get(detected_lang, std_crop)
            extracted["crop_display"] = display_name
            extracted["entities_found"].append(f"Crop: {std_crop} ({display_name})")
            break

    # 2. Match State
    sorted_state_syns = sorted(STATE_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)
    for syn, std_state in sorted_state_syns:
        pattern = r"\b" + re.escape(syn) + r"\b" if syn.isascii() else re.escape(syn)
        if re.search(pattern, clean_text, flags=re.IGNORECASE):
            extracted["state"] = std_state
            display_name = STATE_DISPLAY_MAP.get(std_state, {}).get(detected_lang, std_state)
            extracted["state_display"] = display_name
            extracted["entities_found"].append(f"State: {std_state} ({display_name})")
            break

    # 3. Match Season
    sorted_season_syns = sorted(SEASON_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)
    for syn, std_season in sorted_season_syns:
        pattern = r"\b" + re.escape(syn) + r"\b" if syn.isascii() else re.escape(syn)
        if re.search(pattern, clean_text, flags=re.IGNORECASE):
            extracted["season"] = std_season
            display_name = SEASON_DISPLAY_MAP.get(std_season, {}).get(detected_lang, std_season)
            extracted["season_display"] = display_name
            extracted["entities_found"].append(f"Season: {std_season} ({display_name})")
            break

    # 4. Extract Crop Year
    year_match = re.search(r"\b(199\d|20[0-2]\d)\b", clean_text)
    if year_match:
        extracted["year"] = int(year_match.group(1))
        extracted["entities_found"].append(f"Year: {extracted['year']}")

    # 5. Extract Numbers with contextual keywords
    # Area: first check keyword-then-number, then number-then-unit
    area_match = re.search(
        r"(?:విస్తీర్ణం|area|क्षेत्रफल|रकबा)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        clean_text, flags=re.IGNORECASE
    )
    if not area_match:
        area_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:హెక్టార్లు|హెక్టార్|hectares?|ha|हेक्टेयर|रकबा)",
            clean_text, flags=re.IGNORECASE
        )
    if area_match:
        val = float(area_match.group(1))
        extracted["area"] = val
        extracted["entities_found"].append(f"Area: {val:,.1f} ha")

    # Rainfall: first check keyword-then-number, then number-then-unit
    rain_match = re.search(
        r"(?:వర్షపాతం|rainfall|rain|वर्षा|बारिश)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        clean_text, flags=re.IGNORECASE
    )
    if not rain_match:
        rain_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:మి\.మీ|మిమీ|mm)",
            clean_text, flags=re.IGNORECASE
        )
    if rain_match:
        val = float(rain_match.group(1))
        extracted["rainfall"] = val
        extracted["entities_found"].append(f"Rainfall: {val:,.1f} mm")

    # Fertilizer: keyword-then-number, then number-then-unit
    fert_match = re.search(
        r"(?:ఎరువులు|fertilizer|उर्वरक|खाद)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        clean_text, flags=re.IGNORECASE
    )
    if not fert_match:
        fert_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:ఎరువులు|ఎరువు|fertilizer|उर्वरक|खाद)",
            clean_text, flags=re.IGNORECASE
        )
    if fert_match:
        val = float(fert_match.group(1))
        extracted["fertilizer"] = val
        extracted["entities_found"].append(f"Fertilizer: {val:,.1f}")

    # Fallback positional extraction for remaining unbound numbers
    all_numbers = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", clean_text)]
    # Filter out the year if already identified
    if extracted["year"]:
        all_numbers = [n for n in all_numbers if int(n) != extracted["year"]]

    assigned_vals = {extracted["area"], extracted["rainfall"], extracted["fertilizer"]} - {None}
    unassigned = [n for n in all_numbers if n not in assigned_vals]

    # Heuristic: If user said "వరి ఆంధ్రప్రదేశ్ ఖరీఫ్ 5000 1200", 5000 is area, 1200 is rainfall
    if extracted["area"] is None and unassigned:
        candidate_area = unassigned.pop(0)
        extracted["area"] = candidate_area
        extracted["entities_found"].append(f"Area (inferred): {candidate_area:,.1f} ha")

    if extracted["rainfall"] is None and unassigned:
        candidate_rain = unassigned.pop(0)
        extracted["rainfall"] = candidate_rain
        extracted["entities_found"].append(f"Rainfall (inferred): {candidate_rain:,.1f} mm")

    return extracted


# ── Multilingual Answer & Speech Generation ───────────────────────────────────

def build_multilingual_spoken_answer(
    crop: str,
    predicted_yield: float,
    model_name: str,
    area: float = 0.0,
    lang: str = "en",
) -> Tuple[str, str]:
    """
    Builds the natural spoken answer in the target language (te, hi, en).
    Returns (speech_text, lang_code_for_tts).
    """
    total_prod = predicted_yield * area if area > 0 else 0.0

    if lang == "te":
        crop_te = CROP_DISPLAY_MAP.get(crop, {}).get("te", crop)
        prod_sentence = (
            f"మొత్తం {area:,.0f} హెక్టార్లకు దాదాపు {total_prod:,.0f} టన్నుల ఉత్పత్తి అంచనా వేయబడింది. "
            if area > 0 else ""
        )
        speech_text = (
            f"పంట {crop_te} కోసం అంచనా వేసిన దిగుబడి హెక్టారుకు {predicted_yield:.2f} టన్నులు. "
            f"{prod_sentence}"
            f"ఈ అంచనాను {model_name} మెషిన్ లెర్నింగ్ మోడల్ అందించింది. "
            f"ఇది చారిత్రక వ్యవసాయ డేటా ఆధారిత అంచనా."
        )
        return speech_text, "te-IN"

    if lang == "hi":
        crop_hi = CROP_DISPLAY_MAP.get(crop, {}).get("hi", crop)
        prod_sentence = (
            f"कुल {area:,.0f} हेक्टेयर के लिए लगभग {total_prod:,.0f} टन उत्पादन का अनुमान है। "
            if area > 0 else ""
        )
        speech_text = (
            f"फसल {crop_hi} के लिए अनुमानित उपज {predicted_yield:.2f} टन प्रति हेक्टेयर है। "
            f"{prod_sentence}"
            f"यह अनुमान {model_name} मॉडल द्वारा तैयार किया गया है। "
            f"यह ऐतिहासिक डेटा पर आधारित एक अनुमान है।"
        )
        return speech_text, "hi-IN"

    # Default: English
    prod_sentence = (
        f"Estimated total production for {area:,.0f} hectares is {total_prod:,.0f} metric tonnes. "
        if area > 0 else ""
    )
    speech_text = (
        f"The predicted yield for {crop} is {predicted_yield:.2f} tonnes per hectare. "
        f"{prod_sentence}"
        f"This estimate was computed by the {model_name} model. "
        f"This is an empirical estimate based on historical regional data."
    )
    return speech_text, "en-IN"


def build_multilingual_result_card(
    crop: str,
    predicted_yield: float,
    model_name: str,
    area: float,
    lang: str = "en",
) -> Dict[str, str]:
    """Generates localized result strings for display on the UI."""
    total_prod = predicted_yield * area if area > 0 else 0.0

    if lang == "te":
        crop_name = CROP_DISPLAY_MAP.get(crop, {}).get("te", crop)
        return {
            "title": "🌾 తెలుగులో ఫలిత సారాంశం (Telugu Summary)",
            "headline": f"పంట: **{crop_name}** ({crop})",
            "yield_text": f"అంచనా వేసిన దిగుబడి: **{predicted_yield:.2f} టన్నులు/హెక్టారు**",
            "production_text": f"మొత్తం ఉత్పత్తి ({area:,.0f} ha): **{total_prod:,.0f} టన్నులు**",
            "note": "⚠️ ఇది చారిత్రక డేటా ఆధారిత మోడల్ అంచనా మాత్రమే.",
            "listen_label": "🔊 తెలుగులో సమాధానం వినండి (Listen in Telugu)",
        }

    if lang == "hi":
        crop_name = CROP_DISPLAY_MAP.get(crop, {}).get("hi", crop)
        return {
            "title": "🌾 हिन्दी में परिणाम सारांश (Hindi Summary)",
            "headline": f"फसल: **{crop_name}** ({crop})",
            "yield_text": f"अनुमानित उपज: **{predicted_yield:.2f} टन प्रति हेक्टेयर**",
            "production_text": f"कुल उत्पादन ({area:,.0f} ha): **{total_prod:,.0f} टन**",
            "note": "⚠️ यह ऐतिहासिक डेटा पर आधारित मॉडल अनुमान है।",
            "listen_label": "🔊 हिन्दी में उत्तर सुनें (Listen in Hindi)",
        }

    return {
        "title": "🌾 English Result Summary",
        "headline": f"Crop: **{crop}**",
        "yield_text": f"Predicted Yield: **{predicted_yield:.2f} tonnes/ha**",
        "production_text": f"Total Production ({area:,.0f} ha): **{total_prod:,.0f} tonnes**",
        "note": "⚠️ Empirical estimate based on historical data.",
        "listen_label": "🔊 Listen to spoken response",
    }
