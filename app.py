"""
app.py
------
ADAPT Dashboard — Pure Pretrained API Version
Vision: trpakov/vit-face-expression (HuggingFace)
Text: j-hartmann/emotion-english-distilroberta-base (HuggingFace)
NLG Engine: Gemini 2.5 Flash
"""

import os, re, collections, threading, queue, warnings
import numpy as np
import cv2, av, requests
import streamlit as st
import time
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from transformers import pipeline as hf_pipeline
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import google.generativeai as genai
from stable_baselines3 import PPO
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv() # Loads the .env file locally

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

NUM_EMOTIONS   = 7
EMOTION_NAMES  = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_COLORS = ["#d4574a", "#8e44ad", "#9b59b6", "#f0c040", "#95a5a6", "#5bc0de", "#e67e22"]
EMOTION_EPS    = 0.004
_NEUTRAL_VEC   = [EMOTION_EPS]*4 + [1.0 - 6*EMOTION_EPS] + [EMOTION_EPS]*2

IDX_ANG, IDX_DIS, IDX_FEA = 0, 1, 2
IDX_HAP, IDX_NEU, IDX_SAD, IDX_SUR = 3, 4, 5, 6

ACTION_NAMES = {0:"stay_firm", 1:"small_concession", 2:"medium_concession",
                3:"counter_offer", 4:"accept_deal", 5:"call_bluff", 6:"walk_away"}

WA_INSULT_RATIO, WA_REPEAT_N, WA_NONSENSE_N = 0.30, 3, 4
WA_PATIENCE_START, WA_NEGATIVE_N, WA_STALL_N = 12, 3, 5
ANALYSIS_W, ANALYSIS_H, SKIP_FRAMES = 320, 240, 5

# ==============================================================================
# PRETRAINED MODEL LOADING (Pure HuggingFace)
# ==============================================================================
@st.cache_resource
def load_models():
    print("[ADAPT] Loading Pretrained Vision Model...")
    try:
        vision_pipe = hf_pipeline("image-classification", model="trpakov/vit-face-expression", top_k=7, device=-1, local_files_only=True)
    except Exception:
        vision_pipe = hf_pipeline("image-classification", model="trpakov/vit-face-expression", top_k=7, device=-1)
    
    print("[ADAPT] Loading Pretrained Text Model...")
    try:
        text_pipe = hf_pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7, device=-1, local_files_only=True)
    except Exception:
        text_pipe = hf_pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7, device=-1)
    
    return vision_pipe, text_pipe

@st.cache_resource
def load_rl_model():
    # Looks for a local RL strategy file. If missing, automatically uses rule-based fallback.
    if os.path.exists("phase2_brain.zip"):
        return PPO.load("phase2_brain.zip")
    return None

# ==============================================================================
# INFERENCE HELPERS
# ==============================================================================
def _safe_vec(v) -> np.ndarray:
    arr = np.nan_to_num(np.array(v, dtype="float32"), nan=EMOTION_EPS)
    arr = np.maximum(arr, EMOTION_EPS)
    s = arr.sum()
    return arr / s if s > 0 else np.array(_NEUTRAL_VEC, dtype="float32")

def face_emotion_from_frame(bgr: np.ndarray, vision_pipe) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = vision_pipe(rgb)
    vec = np.full(7, EMOTION_EPS, dtype="float32")
    mapping = {"anger":0, "angry":0, "disgust":1, "fear":2, "joy":3, "happy":3, "neutral":4, "sadness":5, "sad":5, "surprise":6}
    for r in results:
        label = r["label"].lower()
        if label in mapping: vec[mapping[label]] = float(r["score"])
    return _safe_vec(vec)

def text_emotion_from_message(text: str, text_pipe) -> np.ndarray:
    if not text: return _safe_vec(_NEUTRAL_VEC)
    results = text_pipe(text[:512])[0]
    vec = np.full(7, EMOTION_EPS, dtype="float32")
    mapping = {"anger":0, "disgust":1, "fear":2, "joy":3, "neutral":4, "sadness":5, "surprise":6}
    for r in results:
        label = r["label"].lower()
        if label in mapping: vec[mapping[label]] = float(r["score"])
    return _safe_vec(vec)

def compute_bluff_score(face_vec: np.ndarray, text_vec: np.ndarray) -> float:
    # Bluff detection mathematically isolated: A calm face but a highly angry/disgusted text
    face_calm = float(face_vec[IDX_HAP] + face_vec[IDX_NEU])
    text_hostile = float(text_vec[IDX_ANG] + text_vec[IDX_DIS])
    score = max(0.0, (text_hostile * 1.4) - float(face_vec[IDX_ANG]) + (face_calm * 0.4))
    return float(np.clip(score, 0.0, 1.0))

# ==============================================================================
# WEBRTC CAMERA 
# ==============================================================================
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class EmotionAnalyzer(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.latest_emotions = _safe_vec(_NEUTRAL_VEC)
        self.dominant_emotion = "Neutral"
        self.raw_frame = None
        self._frame_q = queue.Queue(maxsize=1)
        self._frame_count = 0
        self._stop = threading.Event()
        self._vm, _ = load_models()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try: 
                bgr = self._frame_q.get(timeout=0.5)
            except queue.Empty: 
                continue
            try:
                vec = face_emotion_from_frame(bgr, self._vm)
                dom = EMOTION_NAMES[int(np.argmax(vec))]
                with self.lock:
                    self.latest_emotions = vec
                    self.dominant_emotion = dom
                    self.raw_frame = bgr.copy()
            except Exception: 
                pass
            
            # Throttle ViT to ~3-4 inferences/sec to free CPU for real-time video
            time.sleep(0.25)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Flip horizontally (1 = mirror mode for natural webcam preview)
        img = cv2.flip(img, 1)

        self._frame_count += 1
        if self._frame_count % SKIP_FRAMES == 0:
            small = cv2.resize(img, (ANALYSIS_W, ANALYSIS_H), interpolation=cv2.INTER_AREA)
            # Ensure the queue always holds the freshest frame, dropping older ones
            if not self._frame_q.empty():
                try: 
                    self._frame_q.get_nowait()
                except queue.Empty: 
                    pass
            try: 
                self._frame_q.put_nowait(small)
            except queue.Full: 
                pass

        with self.lock: 
            dom = self.dominant_emotion

        c_map = {
            "Happy": (0,200,0), "Angry": (0,0,220), "Disgust": (128,0,128),
            "Fear": (0,128,220), "Sad": (200,100,0), "Surprise": (0,180,220), 
            "Neutral": (180,180,180)
        }
        cv2.putText(img, f"LIVE: {dom.upper()}", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c_map.get(dom, (255,255,255)), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def stop(self): 
        self._stop.set()

# ==============================================================================
# INTENT & NLG ENGINE (Gemini 2.5)
# =============================================================================
def classify_intent(text: str) -> str:
    if not GEMINI_API_KEY:
        return "general"
    try:
        r = genai.GenerativeModel("gemini-2.5-flash").generate_content(
            f'Classify into exactly one of [greeting,question,price_offer,accept,reject,general]. Message: "{text}". Reply with ONLY the intent word.'
        )
        intent = r.text.strip().lower()
        return intent if intent in ["greeting", "question", "price_offer", "accept", "reject", "general"] else "general"
    except Exception:
        return "general"

def extract_price(text: str):
    if not text:
        return None
    patterns = [
        r"\$\s*([\d,]+(?:\.\d+)?)",
        r"([\d,]+(?:\.\d+)?)\s*\$",
        r"([\d,]+(?:\.\d+)?)\s*(?:dollars?|bucks?)",
        r"(?:offer|pay|take|give|do|about|for|at)\s+(?:only\s+)?\$?\s*([\d,]+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "").strip()
            try:
                val = float(raw)
                if val > 0:
                    return val
            except ValueError:
                continue
    return None

class NLPEngine:
    def __init__(self):
        self._gemini = genai.GenerativeModel("gemini-2.5-flash") if GEMINI_API_KEY else None

    def generate(self, action: int, face_vec: np.ndarray, fused_vec: np.ndarray, sp: float, ctx: dict) -> str:
        name = ACTION_NAMES.get(action, "stay_firm")
        product = ctx.get("product_name", "the item")
        lp = ctx.get("listing_price", sp)
        offered = ctx.get("buyer_offered")
        intent = ctx.get("intent", "general")
        feats = ctx.get("key_features", [])
        desc = ctx.get("description", "")
        
        face_dom = EMOTION_NAMES[int(np.argmax(face_vec))]
        fused_dom = EMOTION_NAMES[int(np.argmax(fused_vec))]
        feat_s = f"Features: {', '.join(feats[:3])}." if feats else desc[:80] if desc else ""
        off_s = f"${offered:.0f}" if offered else "none"

        if self._gemini:
            try:
                prompt = (
                    f'You are a professional seller negotiating "{product}" (listed at ${lp:.0f}). '
                    f'Buyer intent: {intent}. Buyer offer: {off_s}. '
                    f'Buyer facial expression: {face_dom}, overall psychological mood: {fused_dom}. '
                    f'Your chosen negotiation strategy: {name}. '
                    f'Your current target price: ${sp:.0f}. {feat_s} '
                    f'Respond in 1-2 conversational sentences directly to the buyer. Do not include markdown or quotes.'
                )
                res = self._gemini.generate_content(prompt)
                if res and res.text:
                    return res.text.strip().replace('"', '')
            except Exception as e:
                print(f"[NLPEngine] Gemini NLG fallback: {e}")
                
        if action == 4:
            return f"Deal! I accept your offer of ${sp:.0f}. It's a pleasure doing business with you!"
        elif action == 6:
            return "I don't think we can find a mutually agreeable price today. I will have to pass."
        elif action == 5:
            return f"I appreciate your offer, but I know what this is worth. My price stands at ${sp:.0f}."
        return f"My counter offer is ${sp:.0f}."

# ==============================================================================
# WEB SCRAPER
# ==============================================================================
def scrape_product_url(url: str) -> dict:
    info = {"url": url, "title": "", "price": None, "description": "", "key_features": [], "error": None}
    if not url or not url.startswith(("http://", "https://")):
        info["error"] = "Invalid URL: Must start with http:// or https://"
        return info
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for sel in ["h1", '[class*="product-title"]', '[id*="title"]', "title"]:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                info["title"] = el.get_text(strip=True)[:120]
                break
        for pat in [r'\$\s*[\d,]+\.?\d*']:
            m = re.search(pat, resp.text)
            if m:
                nums = re.findall(r'[\d,.]+', m.group())
                if nums:
                    try:
                        p_val = float(nums[0].replace(',', ''))
                        if p_val > 5.0:
                            info["price"] = p_val
                            break
                    except Exception:
                        pass
        meta = soup.find("meta", attrs={"name": "description"})
        info["description"] = (meta["content"][:300] if meta and meta.get("content") else " ".join(p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 60)[:300])
        info["key_features"] = [li.get_text(strip=True) for li in soup.find_all("li") if 10 < len(li.get_text(strip=True)) < 120][:8]
    except Exception as e:
        info["error"] = str(e)
    return info

# ==============================================================================
# RL NEGOTIATION AGENT 
# ==============================================================================
class RLNegotiationAgent:
    MAX_TURNS = 25

    def __init__(self, rl_model):
        self.rl_model = rl_model
        self.nlp = NLPEngine()
        self.reset(200, 120, "Item")

    def reset(self, listing_price, min_acceptable, product_name, product_info=None):
        self.product_name = product_name
        self.product_info = product_info or {}
        self.state = {
            "turn": 0,
            "listing_price": float(listing_price),
            "seller_price": float(listing_price),
            "min_acceptable": float(min_acceptable),
            "buyer_price": None,
            "total_concessions_made": 0,
            "patience": WA_PATIENCE_START,
            "nonsense_streak": 0,
            "negative_streak": 0,
            "stall_turns": 0,
            "rounds_since_concession": 0,
            "last_buyer_price": None,
            "offer_history": collections.deque(maxlen=WA_REPEAT_N),
            "price_history": [float(listing_price)],
            "negotiation_over": False,
            "deal_made": False,
            "walkaway_reason": None,
            "face_emotion_history": [],
            "fused_emotion_history": [],
            "bluff_history": [],
            "actions": []
        }

    def _get_observation(self, emotion_vec: np.ndarray, bluff_prob: float) -> np.ndarray:
        s = self.state
        norm = lambda x: float(np.clip(x / max(s["listing_price"], 1e-8), 0.0, 1.0))
        padded = np.zeros(5, dtype=np.float32)
        hist = [norm(p) for p in s["price_history"]]
        padded[:min(5, len(hist))] = hist[-5:]
        
        current_buyer = s["buyer_price"] if s["buyer_price"] is not None else (s["listing_price"] * 0.55)
        gap_norm = norm(s["seller_price"] - current_buyer)
        turn_norm = min(float(s["turn"]) / self.MAX_TURNS, 1.0)
        patience_norm = float(np.clip(s["patience"] / WA_PATIENCE_START, 0.0, 1.0))
        product_score = float(np.clip(np.log10(max(s["listing_price"], 1.0)) / np.log10(5000.0), 0.0, 1.0))

        obs = np.concatenate([
            [norm(s["seller_price"])],
            [turn_norm],
            emotion_vec,
            padded,
            [gap_norm],
            [patience_norm],
            [0.5],
            [min(s["rounds_since_concession"] / 5.0, 1.0)],
            [min(s["total_concessions_made"] / 6.0, 1.0)],
            [float(np.clip(bluff_prob, 0.0, 1.0))],
            [min(s["nonsense_streak"] / WA_NONSENSE_N, 1.0)],
            [min(s["stall_turns"] / WA_STALL_N, 1.0)],
            [min(s["negative_streak"] / WA_NEGATIVE_N, 1.0)],
            [product_score]
        ]).astype(np.float32)
        return obs

    def _check_walkaway(self, buyer_offered, face_vec):
        s = self.state
        lp = s["listing_price"]
        
        if s["turn"] >= self.MAX_TURNS:
            return True, f"Maximum turns reached ({self.MAX_TURNS}) without agreement."
            
        if buyer_offered is not None and buyer_offered < lp * WA_INSULT_RATIO:
            return True, f"${buyer_offered:.0f} is below 30% of asking price."
            
        hist = list(s["offer_history"])
        if buyer_offered is not None and len(hist) >= WA_REPEAT_N and all(abs(p - buyer_offered) < 1.0 for p in hist[-WA_REPEAT_N:]):
            return True, f"Same offer repeated {WA_REPEAT_N}× in a row."
            
        if s["nonsense_streak"] >= WA_NONSENSE_N:
            return True, f"No real offer in {WA_NONSENSE_N} consecutive turns."
            
        if s["patience"] <= 0:
            return True, "Patience exhausted — no progress."
            
        if s["negative_streak"] >= WA_NEGATIVE_N:
            return True, f"Sustained negative emotion for {WA_NEGATIVE_N} turns."
            
        if s["stall_turns"] >= WA_STALL_N:
            return True, f"Buyer price stalled for {WA_STALL_N} turns."
            
        return False, None

    def _fallback_action(self, face_vec, fused_vec, bluff_score, intent, gap):
        if intent == "accept" or gap <= 0.01:
            return 4
        if bluff_score > 0.45 or intent == "question":
            return 5
        if face_vec[IDX_HAP] > 0.45 and gap < 0.08:
            return 4
        conc = self.state["total_concessions_made"]
        if gap > 0.30:
            return 2 if conc < 2 else 3
        if gap > 0.15:
            return 1 if conc < 3 else 0
        return 0

    def process_message(self, user_message: str, face_vec: np.ndarray, fused_vec: np.ndarray, bluff_score: float) -> dict:
        s = self.state
        if s["negotiation_over"]:
            return {
                "response": "The negotiation has ended.",
                "action_name": "accept_deal" if s.get("deal_made") else "walk_away",
                "face_vec": face_vec.tolist(),
                "fused_vec": fused_vec.tolist(),
                "bluff_score": bluff_score,
                "walkaway": not s.get("deal_made"),
                "deal_made": s.get("deal_made", False),
                "reason": s["walkaway_reason"],
                "patience": s["patience"],
                "is_bluff": False
            }

        s["turn"] += 1
        s["face_emotion_history"].append(face_vec.tolist())
        s["fused_emotion_history"].append(fused_vec.tolist())
        s["bluff_history"].append(bluff_score)
        
        intent = classify_intent(user_message)
        buyer_offered = extract_price(user_message)

        if buyer_offered is not None:
            s["offer_history"].append(buyer_offered)
            s["buyer_price"] = buyer_offered
            s["last_buyer_price"] = buyer_offered
            s["nonsense_streak"] = 0
        else:
            if s["buyer_price"] is None:
                s["buyer_price"] = s["listing_price"] * 0.55
            if intent in ("general", "reject") and bluff_score < 0.2:
                s["nonsense_streak"] += 1
            else:
                s["nonsense_streak"] = 0

        # Update emotion and stall streaks
        s["negative_streak"] = s["negative_streak"] + 1 if (face_vec[IDX_ANG] + face_vec[IDX_DIS]) > 0.35 else 0
        if s["last_buyer_price"] is not None and buyer_offered is not None and len(s["offer_history"]) > 1:
            prev = list(s["offer_history"])[-2]
            s["stall_turns"] = s["stall_turns"] + 1 if abs(buyer_offered - prev) < 1.0 else 0
        else:
            s["stall_turns"] = 0

        gap = (s["seller_price"] - s["buyer_price"]) / max(s["listing_price"], 1e-8)
        if gap > 0.25:
            s["patience"] -= 2
        elif gap > 0.10:
            s["patience"] -= 1
        if face_vec[IDX_HAP] > 0.35 and gap < 0.12:
            s["patience"] = min(s["patience"] + 1, WA_PATIENCE_START)

        should_walk, reason = self._check_walkaway(buyer_offered, face_vec)

        # Strategy Selection
        deal_reached = False
        if buyer_offered is not None and buyer_offered >= s["seller_price"] and buyer_offered >= s["min_acceptable"]:
            action = 4
            deal_reached = True
        elif should_walk:
            action = 6
            s["negotiation_over"] = True
            s["walkaway_reason"] = reason
        elif self.rl_model is not None:
            # Policy uses the 24-dim fused psychological state
            action = int(self.rl_model.predict(self._get_observation(fused_vec, bluff_score), deterministic=True)[0])
            if action == 4:
                if s["buyer_price"] is None or s["buyer_price"] < s["min_acceptable"]:
                    action = 0
                else:
                    deal_reached = True
            if intent in ["greeting", "general", "question"] and not buyer_offered and action in [1, 2, 3, 4]:
                action = 0
            if s["turn"] <= 2 and action in [1, 2, 4]:
                action = 0
        else:
            action = self._fallback_action(face_vec, fused_vec, bluff_score, intent, gap)
            if action == 4:
                deal_reached = True

        sp = s["seller_price"]
        if action == 1:
            sp = max(sp * 0.96, s["min_acceptable"])
            s["total_concessions_made"] += 1
            s["rounds_since_concession"] = 0
        elif action == 2:
            sp = max(sp * 0.90, s["min_acceptable"])
            s["total_concessions_made"] += 1
            s["rounds_since_concession"] = 0
        elif action == 3:
            sp = max(sp * 0.985, s["min_acceptable"])
            s["total_concessions_made"] += 1
            s["rounds_since_concession"] = 0
        elif action == 4 or deal_reached:
            action = 4
            sp = s["buyer_price"] if (s["buyer_price"] is not None and s["buyer_price"] >= s["min_acceptable"]) else sp
            s["deal_made"] = True
            s["negotiation_over"] = True
            s["rounds_since_concession"] += 1
        else:
            s["rounds_since_concession"] = s.get("rounds_since_concession", 0) + 1

        s["seller_price"] = sp
        s["price_history"].append(sp)

        response = self.nlp.generate(
            action, face_vec, fused_vec, sp,
            {
                "product_name": self.product_name,
                "intent": intent,
                "buyer_offered": buyer_offered,
                "listing_price": s["listing_price"],
                "concessions_made": s["total_concessions_made"],
                "key_features": self.product_info.get("key_features", []),
                "description": self.product_info.get("description", "")
            }
        )
        s["actions"].append(action)

        return {
            "response": response,
            "action_name": ACTION_NAMES.get(action, "stay_firm"),
            "face_vec": face_vec.tolist(),
            "fused_vec": fused_vec.tolist(),
            "bluff_score": bluff_score,
            "walkaway": action == 6,
            "deal_made": s.get("deal_made", False),
            "reason": reason if action == 6 else None,
            "patience": s["patience"],
            "is_bluff": action == 5 and bluff_score > 0.45
        }

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def radar_chart(vec, title="", color="#c9a05c"):
    v=_safe_vec(vec).tolist(); vals=v+[v[0]]; names=EMOTION_NAMES+[EMOTION_NAMES[0]]
    fig=go.Figure(go.Scatterpolar(r=vals,theta=names,fill="toself",fillcolor="rgba(201,160,92,0.10)",line=dict(color=color,width=2.5)))
    fig.update_layout(title=dict(text=title,font=dict(color="#a89880",size=10)),polar=dict(bgcolor="rgba(20,17,14,0.7)",radialaxis=dict(visible=True,range=[0,1],tickvals=[.25,.5,.75],ticktext=["25","50","75"],tickfont=dict(color="#555",size=7)),angularaxis=dict(tickfont=dict(color="#c9a05c",size=10))),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,margin=dict(l=45,r=45,t=30,b=25),height=255)
    return fig

def bar_chart_7(face_vec, fused_vec):
    fp=[round(v*100,1) for v in _safe_vec(face_vec)]; mp=[round(v*100,1) for v in _safe_vec(fused_vec)]
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Face",x=EMOTION_NAMES,y=fp,marker_color=EMOTION_COLORS,text=[f"{p:.1f}%" for p in fp],textposition="outside"))
    fig.add_trace(go.Bar(name="Fused",x=EMOTION_NAMES,y=mp,marker_color=["rgba(255,255,255,0.2)"]*7,marker_line=dict(color=EMOTION_COLORS,width=2),text=[f"{p:.1f}%" for p in mp],textposition="outside"))
    fig.update_layout(barmode="group",xaxis=dict(showgrid=False),yaxis=dict(range=[0,115],showticklabels=False,showgrid=False),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",legend=dict(font=dict(color="#a89880",size=9),bgcolor="rgba(0,0,0,0)"),margin=dict(l=5,r=5,t=5,b=5),height=220)
    return fig

def bluff_gauge(score):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=round(score*100,1),number=dict(suffix="%",font=dict(color="#c9a05c",size=22)),gauge=dict(axis=dict(range=[0,100]),bar=dict(color="#c9a05c"),steps=[dict(range=[0,40],color="rgba(60,200,60,0.12)"),dict(range=[40,65],color="rgba(255,165,0,0.12)"),dict(range=[65,100],color="rgba(200,50,50,0.18)")],threshold=dict(line=dict(color="#d4574a",width=2),value=65)),title=dict(text="🎭 Bluff Score",font=dict(color="#a89880",size=10))))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=20,r=20,t=30,b=10),height=175)
    return fig

# ==============================================================================
# MAIN STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="ADAPT Pretrained", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.stApp{background:#0f0c0a!important;color:#e8dcc8;font-family:monospace}.wa-box{background:#1e0808;border:1px solid #d4574a;border-radius:6px;padding:8px 12px;margin-top:5px;font-size:13px;color:#d4574a}.bluff-box{background:#1a1400;border:1px solid #c9a05c;border-radius:6px;padding:6px 11px;margin-top:3px;font-size:12px;color:#c9a05c}.pat-bar{height:5px;border-radius:3px;background:#222;margin:3px 0 10px}.pat-fill{height:100%;transition:width .4s}</style>""", unsafe_allow_html=True)

def _ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

_ss("agent", None); _ss("messages", []); _ss("started", False); _ss("face_vec", _NEUTRAL_VEC); _ss("fused_vec", _NEUTRAL_VEC); _ss("bluff_score", 0.0); _ss("last_debug", {}); _ss("product_info", {}); _ss("pending_input",None)

vm_pipe, text_pipe = load_models()
rl_model = load_rl_model()

if st.session_state.agent is None: st.session_state.agent = RLNegotiationAgent(rl_model)

_pending = st.session_state.get("pending_input")
if _pending and st.session_state.get("started"):
    with st.spinner("🧠 ADAPT extracting HF semantics..."):
        _text = _pending["text"]
        _face = np.array(_pending["face_vec"], dtype="float32")
        
        # Pure HF Pipeline inference
        _text_vec = text_emotion_from_message(_text, text_pipe)
        
        # 60% Vision / 40% Text Mathematical Fusion
        _fused = _safe_vec((_face * 0.6) + (_text_vec * 0.4)) 
        
        _bluff = compute_bluff_score(_face, _text_vec)
        _result = st.session_state.agent.process_message(_text, _face, _fused, _bluff)
        
    st.session_state.fused_vec, st.session_state.bluff_score, st.session_state.last_debug = _fused.tolist(), _bluff, _result
    st.session_state.messages.append({
        "role": "assistant",
        "content": _result["response"],
        "action": _result["action_name"],
        "walkaway": _result.get("walkaway", False),
        "deal_made": _result.get("deal_made", False),
        "reason": _result.get("reason"),
        "is_bluff": _result.get("is_bluff", False)
    })
    st.session_state.pending_input = None

st.title("🧠 ADAPT — HF Pretrained Model Edition")
st.markdown("**Active Stack:** `trpakov/vit-face-expression` (Vision) • `distilroberta-base` (Text) • `Gemini Flash` (NLG)")

with st.expander("⚙️ Setup", expanded=not st.session_state.started):
    tab_m, tab_u = st.tabs(["📝 Manual", "🔗 Product URL"])
    with tab_m:
        c1, c2, c3 = st.columns(3)
        lp = c1.number_input("Listing Price ($)", value=200, min_value=1)
        ma = c2.number_input("Min Acceptable ($)", value=120, min_value=1)
        pn = c3.text_input("Product Name", "Wireless Headphones")
        if st.button("🚀 Start Negotiation"):
            st.session_state.agent.reset(lp, ma, pn)
            st.session_state.messages = [{"role": "assistant", "content": f"Welcome! {pn} listed at ${lp:.0f}. What's your offer?", "action": "neutral", "walkaway": False, "deal_made": False, "reason": None, "is_bluff": False}]
            for k, v in [("started", True), ("face_vec", _NEUTRAL_VEC), ("fused_vec", _NEUTRAL_VEC), ("bluff_score", 0.0), ("product_info", {})]:
                st.session_state[k] = v
            st.rerun()
    with tab_u:
        url_in = st.text_input("Product URL", "", placeholder="https://…")
        lp_ov = st.number_input("Override Price ($, 0=auto)", value=0, min_value=0)
        if st.button("🌐 Fetch & Start"):
            with st.spinner("Scraping…"):
                info = scrape_product_url(url_in)
            pname, ulp = info.get("title") or "Product", float(lp_ov) if lp_ov > 0 else (float(info["price"]) if info.get("price") else 200.0)
            uma = round(ulp * 0.68, 2)
            st.session_state.product_info = info
            st.session_state.agent.reset(ulp, uma, pname, product_info=info)
            st.session_state.messages = [{"role": "assistant", "content": f"Welcome! {pname} at ${ulp:.0f}. {info.get('description','')[:60]} What's your offer?", "action": "neutral", "walkaway": False, "deal_made": False, "reason": None, "is_bluff": False}]
            for k, v in [("started", True), ("face_vec", _NEUTRAL_VEC), ("fused_vec", _NEUTRAL_VEC), ("bluff_score", 0.0), ("pending_input", None)]:
                st.session_state[k] = v
            st.rerun()

col_chat, col_brain = st.columns([1, 1])

with col_chat:
    st.subheader("💬 Chat")
    ctx_cam = webrtc_streamer(
        key="cam",
        video_processor_factory=EmotionAnalyzer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 480, "max": 640},
                "height": {"ideal": 360, "max": 480},
                "frameRate": {"ideal": 15, "max": 20},
            },
            "audio": False,
        },
        async_processing=True,
    )
    live_face = _NEUTRAL_VEC
    if ctx_cam and ctx_cam.video_processor:
        with ctx_cam.video_processor.lock:
            live_face = list(ctx_cam.video_processor.latest_emotions)
        st.session_state.face_vec = live_face

    for msg in st.session_state.messages:
        pref, atag = "**You:**" if msg["role"] == "user" else "**ADAPT:**", f" `{msg.get('action','')}`" if msg["role"] == "assistant" and msg.get("action") else ""
        st.markdown(f"{pref} {msg['content']}{atag}")
        if msg.get("deal_made"):
            st.markdown(f'<div style="background:#081e08;border:1px solid #2ecc71;border-radius:6px;padding:8px 12px;margin-top:5px;font-size:13px;color:#2ecc71;">🎉 Deal Agreed! Price: ${s["seller_price"]:.2f}</div>', unsafe_allow_html=True)
        if msg.get("walkaway") and msg.get("reason"):
            st.markdown(f'<div class="wa-box">⚠️ Walk-away: {msg["reason"]}</div>', unsafe_allow_html=True)
        if msg.get("is_bluff"):
            st.markdown('<div class="bluff-box">🎭 Bluff detected</div>', unsafe_allow_html=True)

    if st.session_state.agent and st.session_state.agent.state.get("negotiation_over"):
        if st.session_state.agent.state.get("deal_made"):
            st.success("🎉 Negotiation successfully completed — Deal Closed!")
        else:
            st.error("⚠️ Negotiation concluded — Seller has walked away.")
    else:
        user_input = st.chat_input("Make your offer or ask a question…")
        if user_input and st.session_state.started:
            st.session_state.messages.append({"role": "user", "content": user_input, "action": None, "walkaway": False, "deal_made": False, "reason": None, "is_bluff": False})
            st.session_state.pending_input = {"text": user_input, "face_vec": list(live_face)}
            st.rerun()
        elif user_input:
            st.warning("Start a negotiation first.")

with col_brain:
    st.subheader("🧠 XAI Brain")
    s = st.session_state.agent.state
    pat, pct = s["patience"], max(0,int(s["patience"]/WA_PATIENCE_START*100))
    bc = "#c9a05c" if pct>50 else "#e67e22" if pct>25 else "#d4574a"
    st.markdown(f"**Turn:** {s['turn']} | **Price:** ${s['seller_price']:.2f} | **Patience:** {pat}/{WA_PATIENCE_START}")
    st.markdown(f'<div class="pat-bar"><div class="pat-fill" style="width:{pct}%;background:{bc}"></div></div>',unsafe_allow_html=True)
    ca,cb,cc=st.columns(3)
    ca.metric("Nonsense",s["nonsense_streak"],f"/{WA_NONSENSE_N}",delta_color="inverse")
    cb.metric("Negative",s["negative_streak"],f"/{WA_NEGATIVE_N}",delta_color="inverse")
    cc.metric("Stall",s["stall_turns"],f"/{WA_STALL_N}",delta_color="inverse")
    st.plotly_chart(bluff_gauge(st.session_state.bluff_score),use_container_width=True,config={"displayModeBar":False})
    r1,r2=st.columns(2)
    with r1: st.plotly_chart(radar_chart(st.session_state.face_vec,"📸 Face","#5bc0de"),use_container_width=True,config={"displayModeBar":False})
    with r2: st.plotly_chart(radar_chart(st.session_state.fused_vec,"🔀 Fused","#c9a05c"),use_container_width=True,config={"displayModeBar":False})
    st.markdown("#### 📊 7 Emotions")
    st.plotly_chart(bar_chart_7(st.session_state.face_vec,st.session_state.fused_vec),use_container_width=True,config={"displayModeBar":False})
    if dbg := st.session_state.last_debug:
        st.divider(); st.markdown("#### 🔍 Last Decision")
        st.markdown(f"Action: `{dbg.get('action_name','—')}` | Bluff: `{dbg.get('is_bluff',False)}` | Walk: `{dbg.get('walkaway',False)}`")
        if dbg.get("reason"): st.error(dbg["reason"])