"""
Multilingual Support Module
Supports: English, Hindi, Telugu, Tamil, Kannada, Marathi, Bengali
Uses deep-translator (Google Translate, free, no API key needed)
"""

from deep_translator import GoogleTranslator

LANGUAGES = {
    "en": {"name": "English",  "native": "English",  "flag": "🇬🇧"},
    "hi": {"name": "Hindi",    "native": "हिंदी",      "flag": "🇮🇳"},
    "te": {"name": "Telugu",   "native": "తెలుగు",     "flag": "🇮🇳"},
    "ta": {"name": "Tamil",    "native": "தமிழ்",       "flag": "🇮🇳"},
    "kn": {"name": "Kannada",  "native": "ಕನ್ನಡ",      "flag": "🇮🇳"},
    "mr": {"name": "Marathi",  "native": "मराठी",       "flag": "🇮🇳"},
    "bn": {"name": "Bengali",  "native": "বাংলা",       "flag": "🇮🇳"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી",     "flag": "🇮🇳"},
    "pa": {"name": "Punjabi",  "native": "ਪੰਜਾਬੀ",      "flag": "🇮🇳"},
}

# Static translations for UI labels (avoids API calls for common strings)
UI_TRANSLATIONS = {
    "hi": {
        "Crop Advisor":        "फसल सलाहकार",
        "Fertilizer":          "उर्वरक",
        "Disease Scan":        "रोग स्कैन",
        "Market Prices":       "बाजार भाव",
        "Weather":             "मौसम",
        "AI Assistant":        "AI सहायक",
        "Dashboard":           "डैशबोर्ड",
        "Best Pick":           "सर्वोत्तम",
        "Profit":              "मुनाफा",
        "Calculate":           "गणना करें",
        "Analyze":             "विश्लेषण करें",
        "Get Advice":          "सलाह लें",
        "Ask me anything":     "कुछ भी पूछें",
        "Send":                "भेजें",
        "Loading...":          "लोड हो रहा है...",
        "Connected":           "कनेक्टेड",
        "Offline":             "ऑफलाइन",
        "Good Morning":        "शुभ प्रभात",
        "Good Afternoon":      "शुभ दोपहर",
        "Good Evening":        "शुभ संध्या",
    },
    "te": {
        "Crop Advisor":        "పంట సలహాదారు",
        "Fertilizer":          "ఎరువు",
        "Disease Scan":        "వ్యాధి స్కాన్",
        "Market Prices":       "మార్కెట్ ధరలు",
        "Weather":             "వాతావరణం",
        "AI Assistant":        "AI సహాయకుడు",
        "Dashboard":           "డాష్‌బోర్డ్",
        "Best Pick":           "అత్యుత్తమ ఎంపిక",
        "Profit":              "లాభం",
        "Calculate":           "లెక్కించు",
        "Analyze":             "విశ్లేషించు",
        "Get Advice":          "సలహా పొందు",
        "Ask me anything":     "ఏదైనా అడగండి",
        "Send":                "పంపు",
        "Loading...":          "లోడ్ అవుతోంది...",
        "Good Morning":        "శుభోదయం",
        "Good Afternoon":      "శుభ మధ్యాహ్నం",
        "Good Evening":        "శుభ సాయంత్రం",
    },
    "ta": {
        "Crop Advisor":        "பயிர் ஆலோசகர்",
        "Fertilizer":          "உரம்",
        "Disease Scan":        "நோய் ஸ்கேன்",
        "Market Prices":       "சந்தை விலை",
        "Weather":             "வானிலை",
        "AI Assistant":        "AI உதவியாளர்",
        "Good Morning":        "காலை வணக்கம்",
    },
    "kn": {
        "Crop Advisor":        "ಬೆಳೆ ಸಲಹೆಗಾರ",
        "Fertilizer":          "ರಸಗೊಬ್ಬರ",
        "Disease Scan":        "ರೋಗ ಸ್ಕ್ಯಾನ್",
        "Market Prices":       "ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು",
        "Weather":             "ಹವಾಮಾನ",
        "Good Morning":        "ಶುಭೋದಯ",
    },
    "mr": {
        "Crop Advisor":        "पीक सल्लागार",
        "Fertilizer":          "खत",
        "Disease Scan":        "रोग स्कॅन",
        "Market Prices":       "बाजार भाव",
        "Weather":             "हवामान",
        "Good Morning":        "सुप्रभात",
    },
}


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language. Returns original if translation fails."""
    if target_lang == "en" or not text.strip():
        return text
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated or text
    except Exception:
        return text  # fallback to English


def translate_dict(data: dict, target_lang: str, keys_to_translate: list) -> dict:
    """Translate specific keys in a dictionary."""
    if target_lang == "en":
        return data
    result = data.copy()
    for key in keys_to_translate:
        if key in result and isinstance(result[key], str):
            result[key] = translate_text(result[key], target_lang)
    return result


def get_ui_label(key: str, lang: str) -> str:
    """Get translated UI label from static map (fast, no API call)."""
    if lang == "en":
        return key
    lang_map = UI_TRANSLATIONS.get(lang, {})
    return lang_map.get(key, key)


def get_supported_languages() -> list:
    """Return list of all supported languages."""
    return [
        {"code": code, **info}
        for code, info in LANGUAGES.items()
    ]
