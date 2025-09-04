from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import threading
import time

# 載入 .env 變數
load_dotenv()
OCR_API_KEY = os.environ.get("OCR_API_KEY")

class SpamDetector:
    """
    一個封裝所有模型載入和預測邏輯的類別，確保執行緒安全。
    (精簡版：專注於運行 BERT 模型)
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.tokenizer = None
        self.bert_model = None
        self.model_loaded = False
        print("✅ SpamDetector 實例已建立，BERT 模型將在首次請求時載入。")

    def _load_model(self):
        """私有方法，只在需要時載入模型，並確保只執行一次。"""
        if self.model_loaded:
            return
        
        start_time = time.time()
        print("🚀 偵測到首次請求，開始載入 BERT 模型...")
        
        try:
            BERT_MODEL_PATH = './bert_spam_model' 
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            self.bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
            self.model_loaded = True
            print("✅ BERT 模型載入成功")
        except Exception as e:
            print(f"❌ BERT 模型載入失敗：{e}")

        end_time = time.time()
        print(f"👍 模型載入完畢！耗時: {end_time - start_time:.2f} 秒")

    def analyze(self, text):
        """公開方法，執行模型分析。"""
        with self._lock:
            # 確保模型只會被安全地載入一次
            self._load_model()

        if not self.model_loaded or not self.bert_model:
             return {'error': '模型未能成功載入，無法分析'}

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            spam_score = probabilities[0][1].item()
            
            final_label = 'spam' if spam_score >= 0.5 else 'ham'
            
            print(f"原始文字: {text[:50]}...")
            print(f"BERT Score: {spam_score:.4f}")
            
            return {
                'final_label': final_label,
                'text': text,
                'total_score': round(spam_score, 4)
            }
        except Exception as e:
            print(f"❌ BERT 預測錯誤: {e}")
            return {'error': 'BERT 模型預測時發生錯誤'}

# ------------------- 主程式入口 -------------------
app = Flask(__name__)
CORS(app)
detector_instance = SpamDetector()

@app.route('/', methods=['GET'])
def health_check():
    """一個簡單的健康檢查路由，讓 Render 知道服務已啟動。"""
    return "OK", 200

@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    """接收請求，並呼叫 detector 實例進行分析。"""
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
                 extracted_text = result['ParsedResults'][0].get('ParsedText', '').strip()
        
        full_text = f"{extracted_text} {text_input}".strip()

        if image_file and len(extracted_text) < 10:
            return jsonify({'error': '圖片辨識不清，請重新上傳更清晰的圖片'}), 400

        if not full_text: 
            return jsonify({'error': '未提供有效文字'}), 400
        
        analysis_result = detector_instance.analyze(full_text)
        
        if 'error' in analysis_result:
            return jsonify(analysis_result), 500

        return jsonify(analysis_result)

    except Exception as e:
        print(f"❌ 分析時發生未預期錯誤：{e}")
        return jsonify({'error': str(e)}), 500

