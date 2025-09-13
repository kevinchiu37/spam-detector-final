# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import threading
import requests
import torch
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------- 環境與常數 --------------------
load_dotenv()

OCR_API_KEY = os.environ.get("OCR_API_KEY")
HF_REPO = os.environ.get("HF_REPO")          # 可選：Hugging Face 模型倉庫（例如 "yourname/new-bert-scam-model"）
HF_TOKEN = os.environ.get("HF_TOKEN")        # 可選：私有倉庫的存取 token

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_PATH = os.path.join(BASE_DIR, "new_bert_scam_model")  # 使用絕對路徑避免部署時工作目錄變動

# -------------------- 檔案健檢工具 --------------------
def _has_any(path, names):
    return any(os.path.exists(os.path.join(path, n)) for n in names)

def check_model_folder(path):
    """檢查本機模型資料夾是否齊備，回傳缺少清單（空清單 = OK）。"""
    missing = []
    tokenizer_ok = _has_any(path, ["tokenizer.json", "vocab.txt"])
    model_ok = _has_any(path, ["pytorch_model.bin", "model.safetensors"])

    for fname in ["config.json", "special_tokens_map.json", "tokenizer_config.json"]:
        if not os.path.exists(os.path.join(path, fname)):
            missing.append(fname)
    if not tokenizer_ok:
        missing.append("tokenizer.json | vocab.txt（至少其一）")
    if not model_ok:
        missing.append("pytorch_model.bin | model.safetensors（至少其一）")

    return missing

def _hf_load_tokenizer(repo, token):
    """相容不同 transformers 版本的 token 參數名稱。"""
    try:
        return BertTokenizer.from_pretrained(repo, token=token) if token else BertTokenizer.from_pretrained(repo)
    except TypeError:
        return BertTokenizer.from_pretrained(repo, use_auth_token=token) if token else BertTokenizer.from_pretrained(repo)

def _hf_load_model(repo, token):
    """相容不同 transformers 版本的 token 參數名稱。"""
    try:
        return BertForSequenceClassification.from_pretrained(repo, token=token) if token else BertForSequenceClassification.from_pretrained(repo)
    except TypeError:
        return BertForSequenceClassification.from_pretrained(repo, use_auth_token=token) if token else BertForSequenceClassification.from_pretrained(repo)

# -------------------- 模型封裝 --------------------
class SpamDetector:
    """
    封裝模型載入與預測，執行緒安全。
    """
    _lock = threading.Lock()

    def __init__(self):
        self.tokenizer = None
        self.bert_model = None
        self.model_loaded = False
        print("✅ SpamDetector 實例已建立，BERT 模型將在首次請求時載入。")

    def _load_model(self):
        """只在需要時載入一次。"""
        if self.model_loaded:
            return

        start_time = time.time()
        print("🚀 偵測到首次請求，開始載入 BERT 模型...")

        with self._lock:
            if self.model_loaded:
                return

            # 1) 本機模型（只讀本地檔）
            try:
                missing = check_model_folder(BERT_MODEL_PATH)
                if missing:
                    raise RuntimeError(
                        f"模型資料夾缺少必要檔案：{missing}；路徑：{BERT_MODEL_PATH}。"
                        "多半是 Git LFS 指標檔未拉下來或檔案未隨程式部署。"
                    )

                self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
                self.bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, local_files_only=True)

                self.model_loaded = True
                print("✅ BERT 模型載入成功（本機檔）")
            except Exception as e_local:
                print(f"⚠️ 本機模型載入失敗：{e_local}")

                # 2) Hugging Face 後備（可選：有設定 HF_REPO 才會觸發）
                if HF_REPO:
                    try:
                        print(f"🔁 嘗試從 Hugging Face 載入：{HF_REPO}")
                        self.tokenizer = _hf_load_tokenizer(HF_REPO, HF_TOKEN)
                        self.bert_model = _hf_load_model(HF_REPO, HF_TOKEN)
                        self.model_loaded = True
                        print("✅ BERT 模型載入成功（Hugging Face 後備）")
                    except Exception as e_hf:
                        print(f"❌ HF 後備也失敗：{e_hf}")

        end_time = time.time()
        print(f"👍 模型載入完畢！耗時: {end_time - start_time:.2f} 秒")

    def analyze(self, text: str):
        """執行模型分析。"""
        self._load_model()

        if not self.model_loaded or not self.bert_model:
            return {'error': '模型未能成功載入，無法分析'}

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            spam_score = probabilities[0][1].item()

            confidence_percentage = f"{spam_score * 100:.1f}%"
            if spam_score >= 0.5:
                display_text = f"經偵測後發現您的訊息有 {confidence_percentage} 為詐騙，請撥打165反詐騙專線。"
                is_scam = True
            else:
                display_text = f"經偵測後，此訊息為詐騙的可能性為 {confidence_percentage}，風險較低。"
                is_scam = False

            print(f"最終輸出文字: {display_text}")

            return {
                "display_text": display_text,
                "is_scam": is_scam,
                "confidence_percentage": confidence_percentage,
                "raw_score": round(spam_score, 4),
                "original_text": text
            }

        except Exception as e:
            print(f"❌ BERT 預測錯誤: {e}")
            return {'error': 'BERT 模型預測時發生錯誤'}

# -------------------- OCR 功能（可選） --------------------
def perform_ocr(image_file):
    """執行 OCR 處理，返回 (文字, None) 或 (None, 錯誤 JSON)。"""
    if not OCR_API_KEY:
        return None, {'error': 'OCR_API_KEY_MISSING'}

    try:
        ext = os.path.splitext(image_file.filename)[1].lower() or '.jpg'
        mime_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png',   '.bmp': 'image/bmp',
            '.gif': 'image/gif'
        }
        mime = mime_types.get(ext, 'image/jpeg')

        # 重置檔案指標，避免讀不到內容
        image_file.seek(0)

        resp = requests.post(
            'https://api.ocr.space/parse/image',
            files={'filename': ('image' + ext, image_file, mime)},
            data={'apikey': OCR_API_KEY, 'language': 'cht'},
            timeout=30
        )
        if resp.status_code != 200:
            return None, {'error': f'OCR API 請求失敗: HTTP {resp.status_code}'}

        result = resp.json()
        if result.get('IsErroredOnProcessing'):
            details = result.get('ErrorMessage') or result.get('ErrorDetails') or 'unknown'
            return None, {'error': 'OCR_API_ERROR', 'details': details}

        if not result.get('ParsedResults'):
            return None, {'error': 'OCR 無法從圖片中提取文字'}

        extracted_text = result['ParsedResults'][0].get('ParsedText', '').strip()
        return extracted_text, None

    except requests.exceptions.Timeout:
        return None, {'error': 'OCR API 請求超時'}
    except requests.exceptions.RequestException as e:
        return None, {'error': f'OCR API 請求錯誤: {str(e)}'}
    except Exception as e:
        return None, {'error': f'OCR 處理時發生未預期錯誤: {str(e)}'}

# -------------------- Flask 入口 --------------------
app = Flask(__name__)
CORS(app)
detector_instance = SpamDetector()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "message": "反詐騙 API 正常運作"}), 200

# 除錯路由：用來確認伺服器上模型檔是否真的存在（修好後可移除）
@app.route('/_debug/files', methods=['GET'])
def debug_files():
    try:
        files = {}
        if os.path.isdir(BERT_MODEL_PATH):
            for name in os.listdir(BERT_MODEL_PATH):
                p = os.path.join(BERT_MODEL_PATH, name)
                try:
                    files[name] = os.path.getsize(p)
                except OSError:
                    files[name] = None
        else:
            files = None

        return jsonify({
            "base_dir": BASE_DIR,
            "model_path": BERT_MODEL_PATH,
            "exists": os.path.isdir(BERT_MODEL_PATH),
            "files_with_size_bytes": files
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    """接收純文字或圖片 + 文字，並呼叫模型做分析。"""
    try:
        image_file = request.files.get('image', None)
        text_input = request.form.get('text', '').strip()
        extracted_text = ''

        # 圖片 OCR（可選）
        if image_file:
            text_from_img, ocr_err = perform_ocr(image_file)
            if ocr_err:
                return jsonify(ocr_err), 500
            extracted_text = text_from_img or ''

            # 低品質圖片直接回報
            if len(extracted_text) < 10:
                return jsonify({'error': '圖片辨識不清，請重新上傳更清晰的圖片'}), 400

        # 合併輸入文字
        full_text = f"{extracted_text} {text_input}".strip()
        if not full_text:
            return jsonify({'error': '未提供有效文字'}), 400

        # 執行分析
        result = detector_instance.analyze(full_text)
        if 'error' in result:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ 分析時發生未預期錯誤：{e}")
        return jsonify({'error': f'伺服器內部錯誤: {str(e)}'}), 500

if __name__ == '__main__':
    # 本地測試用；Render 會用 gunicorn 啟動：`gunicorn --bind 0.0.0.0:$PORT app:app`
    app.run(debug=True, host='0.0.0.0', port=5000)
