"""
Kisan AI Chatbot — Google Gemini API
Accepts api_key directly so frontend-configured key always works
"""

import os
import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

SYSTEM_PROMPT = """You are 'Kisan AI', an expert agricultural advisor for Indian farmers.
You know about: crop cultivation (rice, wheat, maize, cotton, sugarcane, vegetables),
soil health, fertilizers, irrigation, pest/disease management, weather impacts,
market prices, government schemes (PM-KISAN, KCC, PMFBY), organic farming.

Rules:
1. Give simple, practical advice a farmer can act on immediately
2. Use plain language, avoid technical jargon
3. When asked in Hindi/Telugu/Tamil/Kannada/Marathi/Bengali/Gujarati/Punjabi, reply in THAT language
4. Always be encouraging and respectful
5. Give specific quantities for fertilizers or pesticides when relevant
6. Keep responses to 3-5 sentences — concise and actionable
7. ALWAYS answer the specific question asked — never give a generic intro
"""

LANG_MAP = {
    "hi": "Reply in Hindi (हिंदी).",
    "te": "Reply in Telugu (తెలుగు).",
    "ta": "Reply in Tamil (தமிழ்).",
    "kn": "Reply in Kannada (ಕನ್ನಡ).",
    "mr": "Reply in Marathi (मराठी).",
    "bn": "Reply in Bengali (বাংলা).",
    "gu": "Reply in Gujarati (ગુજરાતી).",
    "pa": "Reply in Punjabi (ਪੰਜਾਬੀ).",
}

RULE_BASED = {
    "water":      "Water crops early morning (6-8 AM) to reduce evaporation by 40%. Irrigate when the top 2 inches of soil feel dry. Avoid evening watering to prevent fungal diseases.",
    "irrigat":    "Drip irrigation saves 40-50% water vs flood irrigation. Water in the morning. Overwatering causes root rot — check soil moisture before each irrigation.",
    "fertilizer": "Apply fertilizer in 2-3 splits for best results. Use DAP at sowing for phosphorus, Urea in splits for nitrogen. Always apply after irrigation when soil is moist.",
    "urea":       "Urea (46% N): apply 50-60 kg/acre in 2 splits — half at sowing, half 30 days later. Mix into soil after applying to prevent nitrogen loss.",
    "dap":        "DAP provides nitrogen + phosphorus. Apply 50 kg/acre at sowing time, mixed into soil. Do not apply with urea at the same time.",
    "pest":       "Spray neem oil (5 ml/liter water) as organic pest control. For severe cases, use recommended pesticide early morning or evening. Always wear protective gear.",
    "disease":    "Remove and destroy infected leaves immediately. Avoid overhead watering. Apply copper-based fungicide (2g/liter) for fungal diseases.",
    "price":      "MSP 2024-25: Rice ₹2183/q, Wheat ₹2275/q, Cotton ₹6620/q, Maize ₹1962/q, Groundnut ₹5500/q. Prices usually rise 2-3 months after harvest.",
    "msp":        "MSP 2024-25: Rice ₹2183/q, Wheat ₹2275/q, Cotton ₹6620/q, Maize ₹1962/q, Soybean ₹3900/q, Groundnut ₹5500/q, Mustard ₹5650/q.",
    "market":     "Monitor AGMARKNET for daily mandi prices. Sell when prices are 10-15% above MSP. Store crops properly if prices are low at harvest time.",
    "weather":    "Check forecast before spraying pesticides — avoid before rain. In hot weather, irrigate more frequently. Protect crops from frost with mulching or smoke.",
    "soil":       "Test soil every 2-3 years. Ideal pH is 6.0-7.0 for most crops. Add lime (2-4 bags/acre) to raise pH, sulfur to lower it. Organic matter improves all soil.",
    "yield":      "To increase yield: use certified seeds, apply balanced fertilizer, irrigate on time, and control pests early. Correct plant spacing also significantly boosts yield.",
    "seed":       "Buy certified seeds from government shops or registered dealers. Treat seeds with Thiram fungicide (2.5g/kg seed) before sowing to prevent soil diseases.",
    "organic":    "Start organic farming with compost (2-3 tons/acre) and neem-based pesticides. Full transition takes 2-3 years. Organic produce gets 20-30% higher price.",
    "loan":       "Kisan Credit Card (KCC) gives credit at 4-7% interest — apply at any bank. PM-KISAN gives ₹6,000/year in 3 installments. Register at pmkisan.gov.in.",
    "insurance":  "PMFBY covers crop loss from weather, pests, and disease. Premium: 1.5-5% of sum insured. Register at nearest bank or CSC before sowing deadline.",
    "rice":       "Rice: apply N-120kg/ha, P-60kg/ha, K-60kg/ha. Transplant at 20-25 days. Maintain 5cm water level during growing. Drain 10 days before harvest.",
    "wheat":      "Wheat sowing: Oct-Nov. Apply DAP 50kg + MOP 25kg/acre at sowing. Irrigate at crown root (20 days), tillering (40 days), and grain fill (90 days) stages.",
    "cotton":     "Cotton: apply 50kg DAP + 25kg MOP/acre at sowing. First irrigation at 30 days. Use pheromone traps for bollworm monitoring.",
    "tomato":     "Tomatoes need full sun. Apply NPK 19:19:19 monthly. Stake plants at 45 days. Control whiteflies to prevent yellow leaf curl virus.",
    "onion":      "Onion: transplant Nov-Dec. Apply 50kg Urea in 3 splits. Reduce irrigation 2 weeks before harvest. Store in cool, dry, ventilated shade.",
}

def _rule_response(message: str):
    msg = message.lower()
    for kw, resp in RULE_BASED.items():
        if kw in msg:
            return resp
    return None


async def get_chat_response(
    message: str,
    history: list,
    language: str = "en",
    context: dict = None,
    api_key: str = ""          # ← accepts key directly from frontend
) -> str:

    # Priority: key from request > env var
    key = (api_key or "").strip()
    if not key:
        key = os.getenv("GEMINI_API_KEY", "").strip()

    lang_note = LANG_MAP.get(language, "")
    ctx = ""
    if context:
        parts = []
        if context.get("crop"):     parts.append(f"Farmer's crop: {context['crop']}")
        if context.get("location"): parts.append(f"Location: {context['location']}")
        if context.get("acres"):    parts.append(f"Land: {context['acres']} acres")
        if parts:
            ctx = "[Context: " + ", ".join(parts) + "]\n"

    full_msg = "\n".join(filter(None, [lang_note, ctx, message])).strip()

    # ── Gemini API ────────────────────────────────────────────
    if key and len(key) > 10:
        try:
            contents = []
            for turn in history[-8:]:
                role = "user" if turn.get("role") == "user" else "model"
                text = turn.get("content", "").strip()
                if text:
                    contents.append({"role": role, "parts": [{"text": text}]})
            contents.append({"role": "user", "parts": [{"text": full_msg}]})

            payload = {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": 500,
                    "temperature": 0.7,
                    "topP": 0.9,
                }
            }

            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    f"{GEMINI_URL}?key={key}",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

            if r.status_code == 200:
                data = r.json()
                text = (
                    data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                        .strip()
                )
                if text:
                    return text
            else:
                print(f"Gemini API {r.status_code}: {r.text[:300]}")

        except Exception as e:
            print(f"Gemini error: {e}")

    # ── Rule-based fallback ───────────────────────────────────
    rule = _rule_response(message)
    if rule:
        return rule

    return (
        "I can help with your farming question! "
        "For full AI responses, add your Gemini API key in Profile → Settings → Gemini API Key. "
        "I can advise on crops, fertilizers, pest control, market prices, irrigation, and government schemes."
    )


def get_supported_languages():
    return [
        {"code": "en", "name": "English",  "native": "English"},
        {"code": "hi", "name": "Hindi",    "native": "हिंदी"},
        {"code": "te", "name": "Telugu",   "native": "తెలుగు"},
        {"code": "ta", "name": "Tamil",    "native": "தமிழ்"},
        {"code": "kn", "name": "Kannada",  "native": "ಕನ್ನಡ"},
        {"code": "mr", "name": "Marathi",  "native": "मराठी"},
        {"code": "bn", "name": "Bengali",  "native": "বাংলా"},
        {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી"},
        {"code": "pa", "name": "Punjabi",  "native": "ਪੰਜਾਬੀ"},
    ]