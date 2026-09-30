# Project ADAPT: Multimodal AI Negotiator

ADAPT is an autonomous, multimodal reinforcement learning agent designed to negotiate pricing in real-time. It processes live WebRTC video feeds and text sentiment to build a 24-dimensional psychological profile of the buyer, using a Proximal Policy Optimization (PPO) strategy engine to maximize seller profit while maintaining user patience.

## 🧠 Architecture Overview

1. **Vision Sensor (Pretrained ViT):** Analyzes live webcam frames to detect 7 distinct facial expressions using a HuggingFace Vision Transformer fine-tuned on FER-2013.
2. **Text Sensor (DistilBERT):** Extracts semantic embeddings from buyer chat messages to calculate text hostility and bluff probability.
3. **Strategy Engine (PPO RL):** A custom Gymnasium environment trained over 80,000 timesteps to select from 7 strategic actions (e.g., small concession, call bluff, walk away) based on real-time emotion vectors.
4. **NLG Engine (Gemini 2.5 Flash):** Translates the RL agent's mathematical action into natural, context-aware dialogue using scraped product specifications.

## 🚀 Local Installation

**1. Clone and Setup Environment**
\`\`\`bash
git clone https://github.com/YourUsername/ADAPT-Negotiator.git
cd ADAPT-Negotiator
python -m venv .venv
# Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Mac/Linux)
\`\`\`

**2. Install Dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**3. Configure API Keys**
Create a `.env` file in the root directory and add your Gemini API Key (required for Natural Language Generation):
\`\`\`text
GEMINI_API_KEY=your_google_ai_studio_key
\`\`\`

## 🎮 Running the Application

To start the local Streamlit dashboard:
\`\`\`bash
streamlit run app.py
\`\`\`

To retrain the Reinforcement Learning policy from scratch:
\`\`\`bash
python train_rl.py
\`\`\`