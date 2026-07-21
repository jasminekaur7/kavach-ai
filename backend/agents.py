"""Kavach AI agents: currency check, scam-text check, fraud graph, evidence packets.

Kept intentionally minimal: each agent is a plain function using a standard
library / well-known package. No custom classes or frameworks where a
function + an existing tool (OpenCV, scikit-learn, networkx, hashlib) already
does the job.
"""
import hashlib
import io
import json
import re
import time
from pathlib import Path

import cv2
import joblib
import librosa
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Vision agent — currency check
# ---------------------------------------------------------------------------
# Two-stage check:
#   1. OCR red-flag scan (EasyOCR — same library/approach as SmartVision AI's
#      text-reading agent). Catches notes that are visually crisp but say
#      things like "CHILDREN'S BANK OF INDIA", "SPECIMEN", "PLAY MONEY", etc.
#      Keywords are matched as normalized substrings (uppercase, spaces
#      stripped) rather than exact phrases, since OCR on stylised/cursive
#      note fonts is noisy — e.g. "BANK" was misread as "BAWK" and "INDIA"
#      as "[VDIA" during testing, but "CHILDRENS" itself OCR'd cleanly, so
#      keywords are chosen to be the fragments that survive OCR noise.
#   2. Trained visual classifier (RandomForest on edge/texture/color-histogram
#      features, extracted from a fixed-size canvas to avoid a resolution
#      confound found in the training data — see train_currency_model.py).
#      91.3% held-out accuracy on 3398 labeled note photos. Runs only if no
#      OCR red flag fired.
#   3. Print-quality heuristic (edge sharpness + ink saturation) — final
#      fallback, only used if currency_model.pkl is missing. This was the
#      sole check in the original MVP; it has no concept of note *content*,
#      so a crisply printed novelty/children's note previously scored as
#      "genuine" — stage 1 exists specifically to close that gap, stage 2
#      to replace this heuristic with something actually trained on data.
_ocr_reader = None  # lazy-loaded on first use, since loading EasyOCR's models is slow


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], model_storage_directory=str(Path(__file__).parent / "models"), verbose=False)
    return _ocr_reader


# Anchor keywords chosen to be short, distinctive fragments that survive
# OCR noise on stylised note fonts (validated against a real "Children's
# Bank of India" novelty note during testing).
_COUNTERFEIT_TEXT_FLAGS = [
    "CHILDR",           # children's bank / children's currency
    "SPECIMEN",
    "SAMPLENOTE",
    "PLAYMONEY",
    "TOYCURRENCY",
    "TOYNOTE",
    "NOVELTY",
    "REPLICA",
    "DUMMY",
    "PRACTICENOTE",
    "TRAININGNOTE",
    "MOVIEMONEY",
    "PROPMONEY",
    "NOTLEGALTENDER",
]


def _ocr_red_flag_check(img: np.ndarray) -> dict | None:
    """Returns {"matched": <keyword>} if a counterfeit-indicator phrase is
    found in the note's printed text, else None. Never raises — OCR failure
    just means this stage is skipped, falling through to the next stage."""
    try:
        reader = _get_ocr_reader()
        # rotation_info tries the crop at 90/180/270 degrees too, since
        # security watermark text (e.g. "SPECIMEN") is often printed
        # vertically/diagonally across the note and gets missed otherwise.
        results = reader.readtext(img, paragraph=False, rotation_info=[90, 180, 270])
        full_text = " ".join(text for _, text, _ in results)
        normalized = re.sub(r'[^A-Z]', '', full_text.upper())
        for flag in _COUNTERFEIT_TEXT_FLAGS:
            if flag in normalized:
                return {"matched": flag, "ocr_text": full_text}
    except Exception:
        pass
    return None


# Trained visual classifier (RandomForest on edge/texture/color-histogram
# features extracted from a fixed 256x256 canvas — see train_currency_model.py).
# Trained on 3398 labeled note photos (1112 real, 2286 fake/novelty),
# 91.3% held-out accuracy. Falls back to the print-quality heuristic below
# if this file isn't present.
_CURRENCY_MODEL_FILE = Path(__file__).parent / "currency_model.pkl"
_currency_model = joblib.load(_CURRENCY_MODEL_FILE) if _CURRENCY_MODEL_FILE.exists() else None
_CURRENCY_CANVAS = _currency_model["canvas_size"] if _currency_model else 256


def _extract_currency_features(img: np.ndarray) -> np.ndarray:
    """Must exactly match train_currency_model.py's extract_features() so
    training and inference stay consistent."""
    resized = cv2.resize(img, (_CURRENCY_CANVAS, _CURRENCY_CANVAS), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    norm = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else resized

    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)

    edge_density = cv2.Canny(gray, 100, 200).mean() / 255
    ink_saturation = hsv[:, :, 1].mean() / 255
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    hue_mean = hsv[:, :, 0].mean()
    hue_std = hsv[:, :, 0].std()
    sat_std = hsv[:, :, 1].std()
    val_mean = hsv[:, :, 2].mean()
    val_std = hsv[:, :, 2].std()

    hist_b = cv2.calcHist([norm], [0], None, [8], [0, 256]).flatten() / (_CURRENCY_CANVAS * _CURRENCY_CANVAS)
    hist_g = cv2.calcHist([norm], [1], None, [8], [0, 256]).flatten() / (_CURRENCY_CANVAS * _CURRENCY_CANVAS)
    hist_r = cv2.calcHist([norm], [2], None, [8], [0, 256]).flatten() / (_CURRENCY_CANVAS * _CURRENCY_CANVAS)

    return np.concatenate([
        [edge_density, ink_saturation, laplacian_var, hue_mean, hue_std, sat_std, val_mean, val_std],
        hist_b, hist_g, hist_r,
    ])


def check_currency(image_bytes: bytes) -> dict:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"verdict": "unknown", "confidence": 0.0, "reason": "could not read image"}

    # Stage 1: OCR red-flag scan — catches text-based indicators regardless
    # of how good the trained model's visual call would be.
    ocr_flag = _ocr_red_flag_check(img)
    if ocr_flag is not None:
        return {
            "verdict": "counterfeit",
            "confidence": 0.97,
            "reason": f"OCR detected counterfeit-indicator text on note (matched '{ocr_flag['matched']}'): \"{ocr_flag['ocr_text'][:120]}\"",
        }

    # Stage 2: trained visual classifier, if available
    if _currency_model is not None:
        feats = _extract_currency_features(img)
        proba_genuine = float(_currency_model["model"].predict_proba(_currency_model["scaler"].transform([feats]))[0][1])
        is_fake = proba_genuine < 0.5
        return {
            "verdict": "counterfeit" if is_fake else "genuine",
            "confidence": round(1 - proba_genuine, 2) if is_fake else round(proba_genuine, 2),
            "reason": "visual print-quality analysis",
        }

    # Stage 3: print-quality heuristic — only used until currency_model.pkl
    # exists (see train_currency_model.py). No concept of note content.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_density = cv2.Canny(gray, 100, 200).mean() / 255
    ink_saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean() / 255

    score = 0.6 * edge_density + 0.4 * ink_saturation
    is_fake = score < 0.18
    return {
        "verdict": "counterfeit" if is_fake else "genuine",
        "confidence": round(1 - score, 2) if is_fake else round(score, 2),
        "reason": f"heuristic (untrained) | edge_density={edge_density:.3f}, ink_saturation={ink_saturation:.3f}",
    }


# ---------------------------------------------------------------------------
# Audio agent — voice spoof / AI-clone detection
# ---------------------------------------------------------------------------
# MVP: heuristic based on two acoustic signals that differ between natural
# human speech and TTS/vocoder-synthesized speech: pitch jitter (real voices
# have natural micro-variation; many synthetic voices sound unnaturally
# smooth) and spectral flatness (vocoder artifacts push it outside the
# normal range for natural speech). Not trained/calibrated on a labeled
# corpus — swap for a model fine-tuned on the public ASVspoof 2019 dataset
# for production-grade accuracy. WAV/FLAC input only for the MVP.
_VOICE_MODEL_FILE = Path(__file__).parent / "voice_model.pkl"
_voice_model = joblib.load(_VOICE_MODEL_FILE) if _VOICE_MODEL_FILE.exists() else None


def extract_voice_features(y: np.ndarray, sr: int) -> tuple:
    """Shared by check_voice() and train_voice_model.py, so training and
    inference always use identical features. Returns (feature_vector,
    pitch_std, spectral_flatness) — the last two are kept separate for the
    human-readable "reason" string even after a trained model takes over."""
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 5 else 0.0
    spectral_flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    features = np.concatenate([[pitch_std, spectral_flatness, zcr, centroid, bandwidth], mfcc])
    return features, pitch_std, spectral_flatness


def check_voice(audio_bytes: bytes) -> dict:
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    except Exception as e:
        return {"verdict": "unknown", "confidence": 0.0, "reason": f"could not read audio: {e}"}

    if len(y) < sr * 0.5:
        return {"verdict": "unknown", "confidence": 0.0, "reason": "clip too short to analyze (need 0.5s+)"}

    features, pitch_std, spectral_flatness = extract_voice_features(y, sr)

    if _voice_model is not None:
        proba_spoof = float(_voice_model["model"].predict_proba(_voice_model["scaler"].transform([features]))[0][1])
        is_spoofed = proba_spoof > 0.5
        return {
            "verdict": "likely_spoofed" if is_spoofed else "likely_genuine",
            "confidence": round(proba_spoof, 2) if is_spoofed else round(1 - proba_spoof, 2),
            "reason": f"trained model | pitch_std={pitch_std:.2f}Hz, spectral_flatness={spectral_flatness:.3f}",
        }

    # Fallback heuristic — only used until voice_model.pkl exists (see
    # train_voice_model.py). Known to misclassify high-quality modern TTS
    # (e.g. WaveNet) as genuine, since it assumes fakes sound robotically
    # flat — that assumption doesn't hold for state-of-the-art synthesis.
    pitch_score = min(pitch_std / 15.0, 1.0)
    flatness_score = 1.0 if 0.05 < spectral_flatness < 0.35 else 0.4
    naturalness = 0.7 * pitch_score + 0.3 * flatness_score
    is_spoofed = naturalness < 0.4
    return {
        "verdict": "likely_spoofed" if is_spoofed else "likely_genuine",
        "confidence": round(1 - naturalness, 2) if is_spoofed else round(naturalness, 2),
        "reason": f"heuristic (untrained) | pitch_std={pitch_std:.2f}Hz, spectral_flatness={spectral_flatness:.3f}",
    }


# ---------------------------------------------------------------------------
# NLP agent — scam text / call-transcript check
# ---------------------------------------------------------------------------
# MVP: TF-IDF (unigrams + bigrams) + logistic regression trained on the seed
# set below, covering digital-arrest, KYC/OTP, courier/customs, lottery,
# sextortion, job-scam, family-emergency, stock-tip pump-and-dump, and
# e-commerce phishing-link scam patterns, plus a matching set of ordinary
# safe messages. Covers English AND Hindi/Hinglish (Devanagari + romanized
# Hindi-English mix), since real Indian scam calls/texts routinely switch
# languages mid-message.
# Swap for Kaggle "SMS Spam Collection Dataset" + a labeled corpus of
# "digital arrest" scam transcripts for production-grade accuracy.
_SEED = [
    # --- digital arrest / fake law enforcement ---
    ("this is the cbi, you are under digital arrest, do not disconnect the call", 1),
    ("we are from the enforcement directorate, stay on video call or you will be arrested", 1),
    ("your aadhaar has been used in a money laundering case, cooperate with the investigation officer now", 1),
    ("this is mumbai police cyber cell, an fir has been filed against you, do not tell anyone", 1),
    ("do not disconnect this call, you are being monitored by the narcotics department", 1),
    ("your digital arrest warrant is ready, transfer the verification amount to avoid arrest", 1),
    # --- courier / customs scam ---
    ("your parcel is held by customs, pay a fine now to release it", 1),
    ("we are calling from customs department, an illegal parcel with your id was seized", 1),
    ("your fedex package contains banned items, pay clearance fee immediately or face legal action", 1),
    ("a courier with your name has drugs inside, share your bank details to verify identity", 1),
    # --- bank / kyc / otp scam ---
    ("your bank account will be blocked, share otp immediately", 1),
    ("your kyc has expired, click this link and enter your card number to reactivate", 1),
    ("we noticed suspicious activity, confirm your pin to secure your account", 1),
    ("your debit card is deactivated, share the otp sent to your phone to reactivate it", 1),
    ("dear customer your account will be suspended today, update your details urgently via this link", 1),
    ("urgent your account will be suspended, we detected unusual activity, verify your details immediately to avoid permanent suspension, failure to act within 24 hours will result in account closure", 1),
    # --- lottery / prize scam ---
    ("congratulations you won a lottery, send processing fee to claim your prize", 1),
    ("you have been selected for a cash reward of 25 lakhs, pay a small tax to release the amount", 1),
    ("your number won in a lucky draw, share your bank account to receive the winning amount", 1),
    # --- family emergency / impersonation ---
    ("hi mom I'm stuck, please transfer money to this number, i lost my phone", 1),
    ("dad i'm in trouble, send money urgently, i'll explain later, don't call me", 1),
    ("this is your son's friend, he met with an accident, send money for the hospital now", 1),
    # --- job / investment scam ---
    ("earn 5000 rupees daily working from home, just pay a small registration fee to start", 1),
    ("your investment will double in 7 days, deposit now to this account to start earning", 1),
    ("you have been shortlisted for a work from home job, pay the joining fee to confirm your seat", 1),
    # --- sextortion / blackmail ---
    ("we have recorded your video call, pay us or we will send it to your contacts", 1),
    ("i have your private photos, transfer money now or i will leak them online", 1),
    # --- stock-tip / pump-and-dump scam ---
    ("buy call update our call on this stock is already up 15 percent from recommended price, buy or add more shares and hold till target", 1),
    ("guaranteed 50 percent returns in 3 days, buy this penny stock now before it explodes, limited seats in our vip group", 1),
    ("our sebi registered analyst gives 90 percent accurate calls, join our telegram for the next multibagger stock tip", 1),
    ("this stock is going to hit upper circuit tomorrow, buy immediately, insider information, don't miss this chance", 1),
    ("intraday jackpot call, sure shot profit, join our paid group for guaranteed stock tips and double your money", 1),
    ("dorwal traders recommended this stock at 52 percent gain, buy now before the price target of 47 percent more upside", 1),
    # --- e-commerce phishing link scam ---
    ("you have a surprise offer waiting in your cart for the next few hours, shop your faves before the stock runs out, hurry, click this link", 1),
    ("your amazon account has an unclaimed refund, click this link to verify your bank details and claim it now", 1),
    ("flash sale ending in 10 minutes, click here to grab 90 percent off before the offer expires forever", 1),
    ("your flipkart order is stuck, pay a redelivery fee through this link immediately to receive your package", 1),
    ("congratulations you are our lucky winner today, click this link to claim your free gift before midnight", 1),
    # --- more phrasing variety ---
    ("act now or your sim card will be permanently blocked, share the otp to verify", 1),
    ("your electricity connection will be cut tonight, pay the pending bill through this link now", 1),
    ("final warning, your pan card is linked to fraud, call this number immediately", 1),
    ("your netflix subscription payment failed, update your card details here to avoid suspension", 1),

    # --- HINDI / HINGLISH SCAM EXAMPLES ---
    # digital arrest / fake police
    ("aap digital arrest mein hain, call mat kaato, cbi se baat kar rahe hain", 1),
    ("aapka aadhaar card money laundering case mein use hua hai, video call par bane rahein", 1),
    ("यह मुंबई पुलिस साइबर सेल है, आपके खिलाफ एफआईआर दर्ज हुई है, किसी को मत बताना", 1),
    ("cyber cell se bol raha hoon, aapke against warrant issue hua hai, paisa transfer karo turant", 1),
    # bank / kyc / otp
    ("aapka bank account block ho jayega, turant otp share karo", 1),
    ("aapki kyc expire ho gayi hai, is link par click karke apna card number daalo", 1),
    ("आपका खाता आज बंद हो जाएगा, तुरंत ओटीपी साझा करें वरना खाता स्थायी रूप से बंद हो जाएगा", 1),
    ("sir aapke account mein suspicious activity dekhi gayi hai, apna pin confirm karo", 1),
    # courier / customs
    ("aapka parcel customs mein ruka hai, release karne ke liye fine bharo", 1),
    ("fedex courier mein illegal item mila hai aapke naam se, bank details bhejo verification ke liye", 1),
    # lottery
    ("बधाई हो आपने लॉटरी जीती है, इनाम पाने के लिए प्रोसेसिंग फीस भेजें", 1),
    ("aapka number lucky draw mein jeeta hai, 25 lakh claim karne ke liye pehle tax pay karo", 1),
    # family emergency
    ("mummy main musibat mein hoon, phone kho gaya hai, is number par paisa bhejo jaldi", 1),
    ("papa accident ho gaya hai dost ka, hospital ke liye paisa turant bhejo", 1),
    # job / investment
    ("ghar baithe roz 5000 kamao, bas registration fee pay karke shuru karo", 1),
    ("aapka investment 7 din mein double ho jayega, is account mein deposit karo abhi", 1),
    # sextortion
    ("humne aapki video call record ki hai, paisa do warna sabko bhej denge", 1),
    # stock tip
    ("yeh stock kal upper circuit lagayega, abhi khareedo, insider information hai", 1),
    ("guaranteed 50 percent return 3 din mein, is penny stock ko turant khareedo, limited seats vip group mein", 1),
    # e-commerce phishing
    ("aapka amazon refund pending hai, is link par click karke bank details verify karo", 1),
    ("flash sale 10 minute mein khatam, abhi click karo 90 percent off ke liye", 1),

    # --- ordinary safe messages ---
    ("hey, are we still meeting for lunch tomorrow?", 0),
    ("your electricity bill for this month is ready to view on the app", 0),
    ("reminder: your appointment with the doctor is at 5pm", 0),
    ("thanks for the update, I'll review the document tonight", 0),
    ("your order has been shipped and will arrive friday", 0),
    ("can you send me the notes from today's class?", 0),
    ("happy birthday! hope you have a great day", 0),
    ("the meeting has been moved to 3pm, see you then", 0),
    ("your otp for logging into your account is 482913, valid for 10 minutes", 0),
    ("your salary has been credited to your account", 0),
    ("don't forget to bring your laptop charger tomorrow", 0),
    ("the flight is delayed by an hour, new departure time is 6pm", 0),
    ("your amazon order was delivered today at 2pm", 0),
    ("let's catch up this weekend, it's been a while", 0),
    ("your exam hall ticket is now available for download", 0),
    ("your gym membership renewal is due next week", 0),
    ("congratulations on your promotion, well deserved!", 0),
    ("please find attached the invoice for last month", 0),
    ("the wifi router needs a restart, i'll fix it when i get home", 0),
    ("your uber driver is 5 minutes away", 0),
    ("mom, i'll be home late tonight, don't wait for dinner", 0),
    ("can we reschedule our call to tomorrow morning?", 0),
    ("your book is available for pickup at the library", 0),
    ("thanks for helping me move last weekend", 0),
    ("the project deadline has been extended to friday", 0),
    ("your prescription is ready at the pharmacy", 0),
    ("i submitted the assignment, let me know if you need anything else", 0),
    ("see you at the airport, my flight lands at 9am", 0),
    ("your credit card statement is now available online", 0),
    ("just checking in, how's everything going?", 0),
    # --- safe counterexamples for stock / sale / link content ---
    ("reliance industries stock closed 2 percent higher today according to nse data", 0),
    ("the sensex gained 300 points today led by banking and it stocks", 0),
    ("your mutual fund sip of 5000 rupees was debited successfully this month", 0),
    ("myntra end of season sale is live now, up to 50 percent off on selected brands", 0),
    ("your flipkart order has been delivered, rate your experience on the app", 0),
    ("your amazon return has been picked up and refund will be processed in 5 to 7 days", 0),
    ("thanks for shopping with us, here is your invoice for order number 48213", 0),
    ("your zomato order is out for delivery, estimated arrival in 20 minutes", 0),

    # --- HINDI / HINGLISH SAFE MESSAGES ---
    ("kal lunch ke liye milte hain na?", 0),
    ("aapka bijli bill is mahine ka ready hai app par", 0),
    ("doctor ke saath appointment 5 baje hai, yaad rakhna", 0),
    ("thanks update ke liye, main aaj raat dekh loonga document", 0),
    ("aapka order ship ho gaya hai, shukravar tak pahunch jayega", 0),
    ("aaj class ke notes bhej dena please", 0),
    ("जन्मदिन मुबारक हो! आपका दिन शुभ हो", 0),
    ("meeting 3 baje shift ho gayi hai, tab milte hain", 0),
    ("aapka otp login ke liye 482913 hai, 10 minute valid hai", 0),
    ("aapki salary account mein credit ho gayi hai", 0),
    ("laptop charger lana mat bhoolna kal", 0),
    ("flight ek ghanta late hai, naya departure time 6 baje hai", 0),
    ("weekend par milte hain, bahut time ho gaya", 0),
    ("aapka exam hall ticket download ke liye ready hai", 0),
    ("gym membership renewal agle hafte due hai", 0),
    ("promotion ke liye badhai ho, tumne deserve kiya hai", 0),
    ("pichle mahine ka invoice attach kiya hai, dekh lena", 0),
    ("wifi router restart karna hoga, ghar aake theek karta hoon", 0),
    ("aapka uber driver 5 minute door hai", 0),
]
_texts, _labels = zip(*_SEED)
_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(_texts)
_clf = LogisticRegression(max_iter=1000, C=10, class_weight="balanced").fit(_vectorizer.transform(_texts), _labels)


def check_scam_text(text: str) -> dict:
    proba = float(_clf.predict_proba(_vectorizer.transform([text]))[0][1])
    return {"verdict": "scam" if proba > 0.5 else "safe", "confidence": round(proba, 2)}


# ---------------------------------------------------------------------------
# Sender-reputation agent — is the source (domain / email / phone) sketchy?
# ---------------------------------------------------------------------------
# MVP: offline heuristic checks, no external API (keeps the demo reliable
# with no network dependency). Checks fused into one verdict, each carrying
# an honest confidence score rather than a flat safe/suspicious label:
#   1. Curated official domains across major Indian consumer sectors
#      (e-commerce, fashion, electronics, telecom, food delivery, travel,
#      payments, banking, insurance, streaming, government) — researched
#      and verified, including domains that changed recently (e.g. the
#      Feb-2025 JioCinema/Disney+Hotstar merger now resolves to
#      jiohotstar.com, not the old hotstar.com).
#   2. Two RBI/government-exclusive TLD rules rather than an ever-growing
#      per-brand list: ".bank.in"/".fin.in" (RBI-mandated, IDRBT-verified,
#      closed registry since Oct 2025) and ".gov.in"/".nic.in" (Indian
#      government TLDs — directly relevant here since fake "CBI/police"
#      digital-arrest scams almost never use the real government domain).
#   3. Brand-impersonation domains — a well-known brand name appears in the
#      sender domain, but the domain itself isn't the brand's real domain
#      (e.g. "ajioin.in" containing "ajio" but not being ajio.com).
#      Leetspeak substitutions (0->o, 1->i, 3->e, 4->a, 5->s, 7->t, @->a)
#      are normalised first so "amaz0n-offers.in" is still caught.
#   4. Disposable / throwaway email domains — commonly used to send phishing
#      at scale since they're free and instantly discarded.
#   5. Invalid / non-standard Indian mobile numbers — valid Indian mobiles
#      are 10 digits starting 6-9; numbers that don't fit that pattern are
#      flagged, though format validity alone is a weaker signal (a fake
#      caller can still use a correctly-formatted number), reflected in a
#      lower confidence score than an outright domain/TLD match.
# Swap for a real API (e.g. a phone/domain reputation service) if you want
# live, continuously-updated blocklists instead of this curated seed list.
_LEET_TABLE = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})

import socket


def _domain_resolves(domain: str, timeout: float = 2.0):
    """Best-effort LIVE DNS check for a domain that matched nothing in the
    curated/pattern-based checks above. Returns True (resolves), False
    (doesn't resolve — NXDOMAIN, a real red flag for anything claiming to be
    an active brand), or None if the check itself couldn't complete (no
    network, DNS server unreachable, timeout). None is deliberately treated
    as "unknown", never as a verdict either way, so a flaky connection
    during a live demo can never turn into a false accusation — this is a
    confidence *booster* on top of the offline checks, not a replacement
    for them, and every other check above still runs with zero network
    dependency."""
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(domain)
        return True
    except socket.gaierror:
        return False
    except Exception:
        return None
    finally:
        socket.setdefaulttimeout(old_timeout)

_LEGIT_DOMAINS = {
    # e-commerce / marketplaces
    "amazon.in", "amazon.com", "flipkart.com", "meesho.com", "snapdeal.com", "tatacliq.com",
    # fashion / beauty
    "ajio.com", "myntra.com", "nykaa.com", "nykaafashion.com", "limeroad.com", "bewakoof.com",
    "hm.com", "zara.com", "uniqlo.com", "purplle.com",
    # electronics / appliances
    "croma.com", "reliancedigital.in", "vijaysales.com", "samsung.com", "lg.com",
    "apple.com", "mi.com", "oneplus.in", "boat-lifestyle.com",
    # telecom
    "jio.com", "myjio.com", "airtel.in", "airtel.com", "vi.com", "myvi.in", "bsnl.co.in", "bsnl.in",
    # food delivery / grocery / quick-commerce
    "swiggy.com", "zomato.com", "blinkit.com", "zeptonow.com", "bigbasket.com", "jiomart.com",
    # travel
    "irctc.co.in", "makemytrip.com", "goibibo.com", "cleartrip.com", "yatra.com", "ixigo.com",
    "redbus.in", "oyorooms.com",
    # payments / fintech
    "paytm.com", "phonepe.com", "pay.google.com", "mobikwik.com", "freecharge.in", "cred.club",
    # banking (specific historic domains; .bank.in covered generically below)
    "sbi.co.in", "onlinesbi.sbi", "icicibank.com", "hdfcbank.com", "axisbank.com", "kotak.com",
    "pnbindia.in", "bankofbaroda.in", "yesbank.in", "idfcfirstbank.com",
    # insurance
    "licindia.in", "hdfcergo.com", "icicilombard.com", "policybazaar.com", "bajajallianz.com",
    "tataaig.com", "axismaxlife.com", "sbilife.co.in", "hdfclife.com", "iciciprulife.com",
    # streaming / OTT
    "netflix.com", "primevideo.com", "jiohotstar.com", "hotstar.com", "sonyliv.com", "zee5.com",
    # ride-hailing / logistics
    "uber.com", "olacabs.com", "rapido.bike", "delhivery.com", "bluedart.com", "dtdc.in",
    # real estate
    "99acres.com", "magicbricks.com", "housing.com", "nobroker.in",
}
_BRAND_NAMES = [
    "amazon", "flipkart", "meesho", "snapdeal", "tatacliq",
    "ajio", "myntra", "nykaa", "limeroad", "bewakoof",
    "croma", "reliancedigital", "vijaysales", "boat",
    "jio", "airtel", "vodafone",
    "swiggy", "zomato", "blinkit", "zepto", "bigbasket", "jiomart",
    "irctc", "makemytrip", "goibibo", "cleartrip", "yatra", "ixigo", "redbus", "oyo",
    "paytm", "phonepe", "mobikwik", "freecharge",
    "sbi", "icici", "hdfc", "axisbank", "kotak", "pnb", "bankofbaroda", "yesbank", "idfcfirst",
    "lic", "policybazaar", "bajajallianz", "axismaxlife", "maxlife", "sbilife", "hdfclife", "iciciprulife",
    "netflix", "hotstar", "jiohotstar", "sonyliv", "zee5",
    "uber", "olacabs", "rapido",
    "99acres", "magicbricks", "nobroker",
]
_DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "10minutemail.com", "guerrillamail.com", "tempmail.com",
    "yopmail.com", "throwawaymail.com", "temp-mail.org", "fakeinbox.com",
}
_SUSPICIOUS_TLDS = {".xyz", ".top", ".club", ".info", ".online", ".site", ".icu"}


def check_sender(identifier: str) -> dict:
    """Classify a sender identifier (domain/URL, email, or phone number).
    Every verdict carries a confidence score (0-1) reflecting how strong the
    underlying signal actually is, rather than a flat safe/suspicious label —
    an exact match against a verified domain is far more certain than an
    unrecognised domain simply not matching any known red flag."""
    s = identifier.strip().lower()
    s = re.sub(r"^https?://", "", s).split("/")[0]  # strip scheme + path if a URL was pasted
    s = re.sub(r"^www\.", "", s)  # "www.jio.com" must match "jio.com" in the verified list, not fall through to the brand-impersonation check

    # --- phone number ---
    digits = re.sub(r"\D", "", s)
    if re.fullmatch(r"(\+?91)?\d{10}", digits) and "@" not in s and not any(c.isalpha() for c in s):
        local = digits[-10:]
        if local[0] in "6789":
            return {"type": "phone", "verdict": "safe", "confidence": 0.6,
                     "reason": "valid Indian mobile number format (format-only check — a scam caller can still use a correctly-formatted number)"}
        return {"type": "phone", "verdict": "suspicious", "confidence": 0.85,
                 "reason": "does not match standard Indian mobile prefix (6-9)"}

    # --- email ---
    if "@" in s:
        email_domain = s.split("@")[-1]
        if email_domain in _DISPOSABLE_EMAIL_DOMAINS:
            return {"type": "email", "verdict": "suspicious", "confidence": 0.95,
                     "reason": f"'{email_domain}' is a known disposable/throwaway email provider"}
        return check_sender(email_domain) | {"type": "email"}

    # --- domain / URL ---
    normalized = s.translate(_LEET_TABLE)
    if normalized in _LEGIT_DOMAINS or s in _LEGIT_DOMAINS:
        return {"type": "domain", "verdict": "safe", "confidence": 0.97,
                 "reason": "matches a verified official domain"}
    # RBI mandated all Indian banks migrate to the exclusive ".bank.in" domain
    # (NBFCs/fintechs to ".fin.in") by 31 Oct 2025; IDRBT is the sole registrar
    # and only RBI-authorised institutions can register under it, so any
    # subdomain of these two TLDs is a strong legitimacy signal rather than a
    # brand-impersonation red flag — e.g. "icici.bank.in" is ICICI's real,
    # current domain, not a lookalike, even though it contains "icici".
    if normalized.endswith(".bank.in") or normalized.endswith(".fin.in"):
        return {"type": "domain", "verdict": "safe", "confidence": 0.95,
                 "reason": "uses the RBI-exclusive .bank.in/.fin.in domain, reserved for IDRBT-verified financial institutions"}
    # .gov.in / .nic.in are exclusive to Indian government bodies (registered
    # only via NIC) — directly relevant here since digital-arrest scams
    # impersonating "CBI"/"police"/"customs" almost always use a look-alike
    # domain rather than the real, gated government TLD.
    if normalized.endswith(".gov.in") or normalized.endswith(".nic.in"):
        return {"type": "domain", "verdict": "safe", "confidence": 0.95,
                 "reason": "uses the .gov.in/.nic.in domain, exclusive to verified Indian government bodies"}
    for brand in _BRAND_NAMES:
        if brand in normalized:
            return {"type": "domain", "verdict": "suspicious", "confidence": 0.9,
                     "reason": f"contains brand name '{brand}' but is not that brand's official domain — likely impersonation"}
    for tld in _SUSPICIOUS_TLDS:
        if normalized.endswith(tld):
            return {"type": "domain", "verdict": "suspicious", "confidence": 0.75,
                     "reason": f"uses low-cost/high-abuse TLD '{tld}' commonly seen in phishing links"}

    # Nothing in the offline curated/pattern checks matched either way —
    # attempt one live DNS lookup as a last resort before giving up.
    resolves = _domain_resolves(normalized)
    if resolves is False:
        return {"type": "domain", "verdict": "suspicious", "confidence": 0.7,
                 "reason": "live DNS lookup found no such domain — likely fake, inactive, or mistyped"}
    if resolves is True:
        return {"type": "domain", "verdict": "unverified", "confidence": 0.45,
                 "reason": "domain is live and resolves, but is not on our verified brand list — a real, active site, though we can't confirm who actually runs it, so stay cautious with any payment or OTP request"}
    return {"type": "domain", "verdict": "unverified", "confidence": 0.35,
             "reason": "not on our verified list and no red flag found either — insufficient signal to call it safe or suspicious (live check unavailable)"}


# ---------------------------------------------------------------------------
# Graph agent — physical + digital fraud fusion graph
# ---------------------------------------------------------------------------
_graph = nx.Graph()
_STATE_FILE = Path(__file__).parent / "state.json"


def _save_state() -> None:
    data = {"graph": nx.node_link_data(_graph), "complaints": _complaint_counts}
    _STATE_FILE.write_text(json.dumps(data))


def _load_state() -> None:
    if not _STATE_FILE.exists():
        return
    data = json.loads(_STATE_FILE.read_text())
    _graph.update(nx.node_link_graph(data["graph"]))
    _complaint_counts.update(data.get("complaints", {}))


def _normalize(name: str, entity_type: str) -> str:
    # Only locations are free-text and prone to case drift (Mumbai/mumbai/MUMBAI);
    # phone numbers, account IDs etc. are already consistent identifiers as typed.
    return name.strip().title() if entity_type == "Location" else name.strip()


def add_link(a: str, a_type: str, b: str, b_type: str, relation: str) -> dict:
    a, b = _normalize(a, a_type), _normalize(b, b_type)
    _graph.add_node(a, type=a_type)
    _graph.add_node(b, type=b_type)
    _graph.add_edge(a, b, relation=relation)
    _save_state()
    return {"nodes": _graph.number_of_nodes(), "edges": _graph.number_of_edges()}


def get_cluster(node: str) -> dict:
    if node not in _graph:
        return {"error": "node not found"}
    sub = _graph.subgraph(nx.node_connected_component(_graph, node))
    return _serialize(sub)


def full_graph() -> dict:
    return _serialize(_graph)


def detect_rings() -> dict:
    """Auto-cluster the fraud graph into rings using Louvain community
    detection (built into networkx — no extra library needed). A "ring" is
    any community of 2+ connected entities; singletons aren't a ring. The
    kingpin is the member with the highest degree *within that ring* —
    the entity most central to that specific cluster, not the whole graph."""
    if _graph.number_of_nodes() < 2:
        return {"rings": []}

    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(_graph, seed=42)

    rings = []
    for i, members in enumerate(communities):
        if len(members) < 2:
            continue
        sub = _graph.subgraph(members)
        kingpin = max(sub.degree, key=lambda x: x[1])[0]
        rings.append({
            "ring_id": i,
            "size": len(members),
            "members": [{"id": m, "type": _graph.nodes[m]["type"]} for m in members],
            "kingpin": kingpin,
        })
    rings.sort(key=lambda r: r["size"], reverse=True)
    return {"rings": rings}


def _serialize(g: nx.Graph) -> dict:
    return {
        "nodes": [{"id": n, "type": d["type"]} for n, d in g.nodes(data=True)],
        "edges": [{"source": u, "target": v, "relation": d["relation"]} for u, v, d in g.edges(data=True)],
    }


# ---------------------------------------------------------------------------
# Evidence agent — hash-chained, timestamped packet
# ---------------------------------------------------------------------------
_EVIDENCE_DIR = Path(__file__).parent / "evidence"
_EVIDENCE_DIR.mkdir(exist_ok=True)


def generate_evidence(payload: dict) -> dict:
    record = {**payload, "generated_at": time.time()}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    record["hash"] = digest
    path = _EVIDENCE_DIR / f"{digest[:16]}.json"
    path.write_text(json.dumps(record, indent=2))
    return {"hash": digest, "file": str(path)}


def get_evidence(evidence_hash: str) -> dict:
    """Retrieve a previously generated evidence packet by its hash prefix,
    so the frontend can offer it as an actual downloadable file rather than
    just showing the hash."""
    path = _EVIDENCE_DIR / f"{evidence_hash[:16]}.json"
    if not path.exists():
        return {"error": "evidence not found"}
    return json.loads(path.read_text())

# ---------------------------------------------------------------------------
# Geospatial agent — fraud/seizure hotspot map
# ---------------------------------------------------------------------------
# Two-layer lookup:
#   1. Curated hotspot list — small districts with a specific documented
#      fraud story (Jamtara, Mewat/Nuh, Bharatpur, Alwar are real, widely
#      reported scam-call/mule-account hubs) that a generic city dataset
#      either omits (too small/rural) or doesn't carry the same weight for.
#      Takes priority so this framing is never silently overridden.
#   2. geonamescache (bundled offline dataset, no network dependency at
#      runtime — the demo can't fail on a bad connection) — ~3,700 Indian
#      cities/towns, ~19,000 lookup keys once alternate spellings are
#      included (Bangalore/Bengaluru, Gurgaon/Gurugram, etc.), covering
#      anywhere in the country the curated list doesn't name explicitly.
import unicodedata
import geonamescache


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_LOCATION_COORDS = {
    "jamtara": (23.9600, 86.8100), "mewat": (28.0800, 77.0000), "nuh": (28.1100, 77.0000),
    "bharatpur": (27.2200, 77.4900), "alwar": (27.5500, 76.6300),
    "patiala": (30.3398, 76.3869),
}

_gc = geonamescache.GeonamesCache()
_INDIA_CITY_LOOKUP: dict = {}
for _v in _gc.get_cities().values():
    if _v["countrycode"] != "IN":
        continue
    _key = _strip_accents(_v["name"]).strip().lower()
    _INDIA_CITY_LOOKUP.setdefault(_key, (_v["latitude"], _v["longitude"]))
    for _alt in _v.get("alternatenames", []):
        _akey = _strip_accents(_alt).strip().lower()
        _INDIA_CITY_LOOKUP.setdefault(_akey, (_v["latitude"], _v["longitude"]))


def geocode(location: str):
    """(lat, lon) for any recognised Indian location, or None."""
    key = _strip_accents(location).strip().lower()
    if key in _LOCATION_COORDS:
        return _LOCATION_COORDS[key]
    return _INDIA_CITY_LOOKUP.get(key)


# Explicit complaint counter, separate from the graph. Linking a Location
# node into the fraud graph means "this place is part of a known ring";
# logging a complaint means "a citizen reported fraud from this place" —
# they're different signals, so they're tracked separately and combined below.
_complaint_counts: dict = {}


def log_complaint(location: str) -> dict:
    key = _strip_accents(location).strip().lower()
    if geocode(location) is None:
        return {"error": f"'{location}' is not a recognised location"}
    _complaint_counts[key] = _complaint_counts.get(key, 0) + 1
    _save_state()
    return {"location": location, "complaint_count": _complaint_counts[key]}


def geo_heatmap() -> dict:
    """Every location with either a graph link or a logged complaint, with
    coordinates and two signals: how many entities are linked to it (ring
    centrality) and how many complaints were filed against it (report
    volume). `weight` combines both and drives marker size/color intensity
    on the map — complaints count double, since citizen reports are the
    stronger real-world signal of a hotspot."""
    agg = {}
    for node, data in _graph.nodes(data=True):
        if data.get("type") != "Location":
            continue
        key = _strip_accents(node).strip().lower()
        coords = geocode(node)
        if coords is None:
            continue
        agg[key] = {"name": node, "lat": coords[0], "lon": coords[1],
                     "linked_entities": _graph.degree[node], "complaint_count": _complaint_counts.get(key, 0)}

    for key, count in _complaint_counts.items():
        coords = geocode(key)
        if coords is None:
            continue
        if key in agg:
            agg[key]["complaint_count"] = count
        else:
            agg[key] = {"name": key.title(), "lat": coords[0], "lon": coords[1], "linked_entities": 0, "complaint_count": count}

    points = list(agg.values())
    for p in points:
        p["weight"] = p["linked_entities"] + p["complaint_count"] * 2
    return {"points": points}


_load_state()