"""
setup_models.py
---------------
Pre-downloads and caches the required HuggingFace pretrained models for ADAPT.
No Kaggle datasets or custom weights are required.
"""

from transformers import pipeline

def download_pretrained_models():
    print("="*60)
    print("ADAPT — Pretrained Model Setup Utility")
    print("="*60)
    print("\nCaching HuggingFace models to local system. This may take a minute...\n")

    try:
        # 1. Vision Model (Facial Expression Recognition)
        print("1/2: Downloading Vision Model (trpakov/vit-face-expression)...")
        pipeline("image-classification", model="trpakov/vit-face-expression", top_k=7, device=-1)
        print("✅ Vision model cached successfully.\n")

        # 2. Text Model (Semantic Emotion Classification)
        print("2/2: Downloading Text Model (j-hartmann/emotion-english-distilroberta-base)...")
        pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7, device=-1)
        print("✅ Text model cached successfully.\n")

        print("="*60)
        print("🎉 All pretrained models are ready! You can now run:")
        print("   streamlit run app.py")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")

if __name__ == "__main__":
    download_pretrained_models()