from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import os
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 載入 .env 變數
load_dotenv()
OCR_API_KEY = os.environ.get("OCR_API_KEY")

app = Flask(__name__)
CORS(app)

# --- 修改: 將模型變數初始化為 None ---
# 我們不再在伺服器啟動時載入模型，而是在第一次請求時才載入
sklearn_model = None
vectorizer = None
tokenizer = None
bert_model = None
models_loaded = False # 新增一個旗標來判斷模型是否已載入

# --- 新增: 建立一個專門用來載入模型的函式 ---
def load_models():
    global sklearn_model, vectorizer, tokenizer, bert_model, models_loaded
    if models_loaded: # 如果已經載入過，就直接返回
        return

    print("🚀 偵測到首次請求，開始載入所有模型...")
    
    # 載入傳統模型 (Scikit-learn)
    try:
        sklearn_model = joblib.load('spam_detector_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        print("✅ 傳統 Scikit-learn 模型載入成功")
    except Exception as e:
        print(f"❌ 傳統 Scikit-learn 模型或向量器載入失敗：{e}")

    # 載入新的 BERT 模型
    try:
        BERT_MODEL_PATH = './bert_spam_model' 
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        print("✅ 新的 BERT 模型載入成功")
    except Exception as e:
        print(f"❌ BERT 模型載入失敗：{e}")

    models_loaded = True
    print("👍 所有模型載入完畢！")
# --- 新增結束 ---


# --- 獨立的 BERT 預測函式 (保持不變) ---
def predict_with_bert(text):
    if not tokenizer or not bert_model:
        return "ham", 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        spam_probability = probabilities[0][1].item()
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        return 'spam' if predicted_class_id == 1 else 'ham', spam_probability
    except Exception as e:
        print(f"❌ BERT 預測錯誤: {e}")
        return "ham", 0.0

# --- 修改: `/analyze-all` API ---
@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    # --- **新增: 在處理請求的一開始，先確保模型已載入** ---
    if not models_loaded:
        load_models()
    # --- 新增結束 ---

    try:
        image_file = request.files.get('image', None)
        text_input = request.form.get('text', '').strip()
        extracted_text = ''

        if image_file:
            if not OCR_API_KEY:
                return jsonify({'error': 'OCR_API_KEY_MISSING'}), 500

            ext = os.path.splitext(image_file.filename)[1].lower() or '.jpg'
            mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.bmp': 'image/bmp', '.gif': 'image/gif'}
            mime = mime_types.get(ext, 'image/jpeg')

            ocr_response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'filename': ('image' + ext, image_file, mime)},
                data={'apikey': OCR_API_KEY, 'language': 'cht'}
            )
            result = ocr_response.json()
            
            if result.get('IsErroredOnProcessing'):
                details = result.get('ErrorMessage') or result.get('ErrorDetails') or 'unknown'
                return jsonify({'error': 'OCR_API_ERROR', 'details': details}), 500
            
            if result.get('ParsedResults'):
                 extracted_text = result['ParsedResults'][0].get('ParsedText', '')
        
        full_text = f"{extracted_text.strip()} {text_input}".strip()

        if image_file and len(extracted_text.strip()) < 10:
            return jsonify({'error': '圖片辨識不清，請重新上傳更清晰的圖片'}), 400

        if not full_text:
            return jsonify({'error': '未提供有效文字'}), 400

        sklearn_score = 0.0
        if sklearn_model and vectorizer:
            vec = vectorizer.transform([full_text])
            sklearn_score = sklearn_model.predict_proba(vec)[0][1]

        bert_label, bert_score = predict_with_bert(full_text)
        
        total_score = (sklearn_score + bert_score) / 2
        
        final_label = 'spam' if total_score >= 0.5 else 'ham'
        
        print(f"原始文字: {full_text[:50]}...")
        print(f"SK-Learn Score: {sklearn_score:.4f}, BERT Score: {bert_score:.4f}, Total Score: {total_score:.4f}")
        
        return jsonify({
            'final_label': final_label,
            'text': full_text,
            'total_score': round(total_score, 4)
        })

    except Exception as e:
        print(f"❌ 分析時錯誤：{e}")
        return jsonify({'error': str(e)}), 500

# (原本的 /predict 和 if __name__ == '__main__' 可以保留或刪除，它們不會影響部署)

