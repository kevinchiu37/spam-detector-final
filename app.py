from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import os
from dotenv import load_dotenv

# --- 新增: 為了載入 BERT 模型 ---
from transformers import BertTokenizer, BertForSequenceClassification
import torch
# ------------------------------------

# 載入 .env 變數
load_dotenv()
OCR_API_KEY = os.environ.get("OCR_API_KEY")
if not OCR_API_KEY:
    print("⚠️ 未設定 OCR_API_KEY，圖片辨識功能將無法使用")

app = Flask(__name__)
CORS(app)

# --- 載入所有模型 ---
# 傳統模型 (Scikit-learn)
try:
    sklearn_model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ 傳統 Scikit-learn 模型載入成功")
except Exception as e:
    print(f"❌ 傳統 Scikit-learn 模型或向量器載入失敗：{e}")
    sklearn_model = None
    vectorizer = None

# 新的 BERT 模型
try:
    BERT_MODEL_PATH = './bert_spam_model' 
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    print("✅ 新的 BERT 模型載入成功")
except Exception as e:
    print(f"❌ BERT 模型載入失敗：{e}")
    tokenizer = None
    bert_model = None
# ------------------------------------

# --- 獨立的 BERT 預測函式 ---
def predict_with_bert(text):
    if not tokenizer or not bert_model:
        return None, 0.0

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
        return None, 0.0
# ------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': '請提供 text 欄位'}), 400

        vec = vectorizer.transform([text])
        pred = sklearn_model.predict(vec)[0]
        return jsonify({'label': 'spam' if pred == 1 else 'ham'})

    except Exception as e:
        print(f"❌ 預測錯誤：{e}")
        return jsonify({'error': str(e)}), 500

# --- 修改: `/analyze-all` API 以融合兩種模型 ---
@app.route('/analyze-all', methods=['POST'])
def analyze_all():
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
        
        # --- 修改: 將驗證邏輯移至正確的位置 ---
        # 1. 先合併所有文字
        full_text = f"{extracted_text.strip()} {text_input}".strip()

        # 2. 如果有上傳圖片，優先檢查圖片辨識品質
        if image_file and len(extracted_text.strip()) < 10:
            return jsonify({'error': '圖片辨識不清，請重新上傳更清晰的圖片'}), 400

        # 3. 最後才檢查是否完全沒有任何文字
        if not full_text:
            return jsonify({'error': '未提供有效文字'}), 400
        # --- 修改結束 ---

        # --- 模型融合邏輯 ---
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
