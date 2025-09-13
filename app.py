from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import threading
import time
import logging
from werkzeug.utils import secure_filename
from functools import lru_cache

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()
OCR_API_KEY = os.environ.get("OCR_API_KEY")

# 設定常數
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
MIN_TEXT_LENGTH = 10
DEFAULT_MODEL_PATH = './new_bert_scam_model'
OCR_TIMEOUT = 30
MAX_TEXT_LENGTH = 512

class SpamDetector:
    """
    單例模式的垃圾訊息偵測器，確保執行緒安全且資源效率最佳化
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """實作 Singleton 模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 避免重複初始化
        if hasattr(self, 'initialized'):
            return
            
        self.tokenizer = None
        self.bert_model = None
        self.model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = True
        self.model_path = os.environ.get('BERT_MODEL_PATH', DEFAULT_MODEL_PATH)
        
        logger.info(f"✅ SpamDetector 實例已建立 (裝置: {self.device})")
        logger.info(f"📁 模型路徑: {self.model_path}")

    def _load_model(self):
        """延遲載入模型，只在需要時載入"""
        if self.model_loaded:
            return True
        
        with self._lock:
            # 雙重檢查鎖定
            if self.model_loaded:
                return True
                
            start_time = time.time()
            logger.info("🚀 開始載入 BERT 模型...")
            
            try:
                # 檢查模型路徑是否存在
                if not os.path.exists(self.model_path):
                    logger.error(f"❌ 模型路徑不存在: {self.model_path}")
                    return False
                
                # 載入 tokenizer 和模型
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.bert_model = BertForSequenceClassification.from_pretrained(
                    self.model_path
                ).to(self.device)
                
                # 設定為評估模式
                self.bert_model.eval()
                self.model_loaded = True
                
                elapsed_time = time.time() - start_time
                logger.info(f"✅ BERT 模型載入成功 (耗時: {elapsed_time:.2f} 秒)")
                return True
                
            except Exception as e:
                logger.error(f"❌ BERT 模型載入失敗：{e}")
                return False

    @lru_cache(maxsize=128)
    def _predict_cached(self, text_hash):
        """快取預測結果以提升相同文字的處理速度"""
        # 注意：實際使用時需要傳入原始文字而非 hash
        pass

    def analyze(self, text, use_cache=True):
        """
        分析文字是否為詐騙訊息
        
        Args:
            text (str): 要分析的文字
            use_cache (bool): 是否使用快取
            
        Returns:
            dict: 分析結果
        """
        # 確保模型已載入
        if not self._load_model():
            logger.error("模型載入失敗")
            return {
                'error': '模型未能成功載入，無法分析',
                'code': 'MODEL_LOAD_ERROR'
            }

        # 文字預處理
        text = text.strip()
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"文字長度 {len(text)} 超過上限，將進行截斷")
            text = text[:MAX_TEXT_LENGTH]

        try:
            # Tokenize 輸入文字
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            ).to(self.device)
            
            # 進行預測
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            # 計算機率
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            spam_score = probabilities[0][1].item()
            
            # 格式化輸出
            confidence_percentage = f"{spam_score * 100:.1f}%"
            
            # 根據閾值判斷
            thresholds = {
                'high': 0.8,
                'medium': 0.5,
                'low': 0.2
            }
            
            if spam_score >= thresholds['high']:
                risk_level = '高'
                display_text = f"⚠️ 高風險警告：此訊息有 {confidence_percentage} 的機率為詐騙。強烈建議撥打 165 反詐騙專線。"
                is_scam = True
            elif spam_score >= thresholds['medium']:
                risk_level = '中'
                display_text = f"⚠️ 中風險提醒：此訊息有 {confidence_percentage} 的機率為詐騙。建議謹慎處理。"
                is_scam = True
            else:
                risk_level = '低'
                display_text = f"✅ 此訊息為詐騙的可能性為 {confidence_percentage}，風險較低。"
                is_scam = False

            result = {
                "success": True,
                "display_text": display_text,
                "is_scam": is_scam,
                "risk_level": risk_level,
                "confidence_percentage": confidence_percentage,
                "raw_score": round(spam_score, 4),
                "text_length": len(text),
                "model_version": "BERT-v1.0",
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"分析完成 - 風險等級: {risk_level}, 分數: {confidence_percentage}")
            return result
            
        except Exception as e:
            logger.error(f"❌ BERT 預測錯誤: {e}", exc_info=True)
            return {
                'error': f'模型預測時發生錯誤: {str(e)}',
                'code': 'PREDICTION_ERROR'
            }

def validate_image(image_file):
    """
    驗證上傳的圖片檔案
    
    Args:
        image_file: Flask 檔案物件
        
    Returns:
        tuple: (是否有效, 錯誤訊息)
    """
    if not image_file:
        return False, "未提供圖片檔案"
    
    # 檢查檔案名稱
    filename = secure_filename(image_file.filename)
    if not filename:
        return False, "無效的檔案名稱"
    
    # 檢查副檔名
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支援的檔案格式。支援格式: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # 檢查檔案大小
    image_file.seek(0, os.SEEK_END)
    file_size = image_file.tell()
    image_file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"檔案大小超過限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)"
    
    return True, None

def perform_ocr(image_file):
    """
    執行 OCR 處理，返回提取的文字
    
    Args:
        image_file: Flask 檔案物件
        
    Returns:
        tuple: (提取的文字, 錯誤字典)
    """
    if not OCR_API_KEY:
        logger.error("OCR_API_KEY 未設定")
        return None, {
            'error': 'OCR 服務未設定',
            'code': 'OCR_API_KEY_MISSING'
        }

    try:
        # 驗證圖片
        is_valid, error_msg = validate_image(image_file)
        if not is_valid:
            return None, {'error': error_msg, 'code': 'INVALID_IMAGE'}
        
        # 準備檔案上傳
        ext = os.path.splitext(image_file.filename)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg', 
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png', 
            '.bmp': 'image/bmp', 
            '.gif': 'image/gif'
        }
        mime = mime_types.get(ext, 'image/jpeg')
        
        # 重置檔案指針
        image_file.seek(0)
        
        # 發送 OCR 請求
        logger.info(f"發送 OCR 請求 (檔案類型: {mime})")
        
        ocr_response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'filename': (f'image{ext}', image_file, mime)},
            data={
                'apikey': OCR_API_KEY,
                'language': 'cht',
                'isOverlayRequired': False,
                'detectOrientation': True,
                'scale': True,
                'OCREngine': 2  # 使用 OCR Engine 2 以獲得更好的中文識別
            },
            timeout=OCR_TIMEOUT
        )
        
        # 檢查 HTTP 狀態碼
        if ocr_response.status_code != 200:
            logger.error(f"OCR API 回應錯誤: HTTP {ocr_response.status_code}")
            return None, {
                'error': f'OCR 服務暫時無法使用 (HTTP {ocr_response.status_code})',
                'code': 'OCR_HTTP_ERROR'
            }
        
        # 解析回應
        result = ocr_response.json()
        
        # 檢查處理錯誤
        if result.get('IsErroredOnProcessing'):
            error_details = (
                result.get('ErrorMessage') or 
                result.get('ErrorDetails') or 
                '未知錯誤'
            )
            logger.error(f"OCR 處理錯誤: {error_details}")
            return None, {
                'error': 'OCR 處理失敗',
                'details': error_details,
                'code': 'OCR_PROCESSING_ERROR'
            }
        
        # 提取文字
        parsed_results = result.get('ParsedResults', [])
        if not parsed_results:
            logger.warning("OCR 無法提取文字")
            return None, {
                'error': '無法從圖片中識別出文字，請確保圖片清晰且包含文字',
                'code': 'NO_TEXT_FOUND'
            }
        
        extracted_text = parsed_results[0].get('ParsedText', '').strip()
        
        # 記錄結果
        logger.info(f"OCR 成功提取 {len(extracted_text)} 個字元")
        
        return extracted_text, None
        
    except requests.exceptions.Timeout:
        logger.error("OCR API 請求超時")
        return None, {
            'error': 'OCR 服務回應超時，請稍後再試',
            'code': 'OCR_TIMEOUT'
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"OCR API 請求錯誤: {e}")
        return None, {
            'error': 'OCR 服務連線失敗',
            'details': str(e),
            'code': 'OCR_REQUEST_ERROR'
        }
    except Exception as e:
        logger.error(f"OCR 處理未預期錯誤: {e}", exc_info=True)
        return None, {
            'error': '處理圖片時發生錯誤',
            'details': str(e),
            'code': 'OCR_UNEXPECTED_ERROR'
        }

# ------------------- Flask 應用程式 -------------------
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'https://yourdomain.com'])  # 限制 CORS 來源

# 設定檔案上傳限制
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 初始化偵測器
detector_instance = SpamDetector()

@app.errorhandler(413)
def request_entity_too_large(error):
    """處理檔案過大錯誤"""
    return jsonify({
        'error': f'檔案大小超過限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)',
        'code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """處理內部伺服器錯誤"""
    logger.error(f"內部伺服器錯誤: {error}")
    return jsonify({
        'error': '伺服器內部錯誤，請稍後再試',
        'code': 'INTERNAL_ERROR'
    }), 500

@app.route('/', methods=['GET'])
def health_check():
    """健康檢查端點"""
    status = {
        "status": "OK",
        "message": "反詐騙 API 正常運作",
        "version": "2.0",
        "model_loaded": detector_instance.model_loaded,
        "ocr_enabled": bool(OCR_API_KEY),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(status), 200

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """純文字分析端點"""
    try:
        # 取得請求資料
        data = request.get_json()
        if not data:
            return jsonify({
                'error': '請提供 JSON 格式的資料',
                'code': 'INVALID_REQUEST'
            }), 400
        
        text = data.get('text', '').strip()
        
        # 驗證文字
        if not text:
            return jsonify({
                'error': '請提供要分析的文字',
                'code': 'NO_TEXT'
            }), 400
        
        if len(text) < MIN_TEXT_LENGTH:
            return jsonify({
                'error': f'文字長度至少需要 {MIN_TEXT_LENGTH} 個字元',
                'code': 'TEXT_TOO_SHORT'
            }), 400
        
        # 執行分析
        result = detector_instance.analyze(text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"分析文字時發生錯誤: {e}", exc_info=True)
        return jsonify({
            'error': '處理請求時發生錯誤',
            'details': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """圖片 OCR 分析端點"""
    try:
        # 檢查是否有圖片
        if 'image' not in request.files:
            return jsonify({
                'error': '請上傳圖片檔案',
                'code': 'NO_IMAGE'
            }), 400
        
        image_file = request.files['image']
        
        # 執行 OCR
        extracted_text, error = perform_ocr(image_file)
        
        if error:
            return jsonify(error), 400
        
        # 檢查提取的文字
        if len(extracted_text) < MIN_TEXT_LENGTH:
            return jsonify({
                'error': f'圖片中識別的文字太少（最少需要 {MIN_TEXT_LENGTH} 個字元），請上傳更清晰的圖片',
                'code': 'INSUFFICIENT_TEXT'
            }), 400
        
        # 執行分析
        result = detector_instance.analyze(extracted_text)
        result['source'] = 'image_ocr'
        result['extracted_text'] = extracted_text[:200] + '...' if len(extracted_text) > 200 else extracted_text
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"分析圖片時發生錯誤: {e}", exc_info=True)
        return jsonify({
            'error': '處理圖片時發生錯誤',
            'details': str(e),
            'code': 'IMAGE_PROCESSING_ERROR'
        }), 500

@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    """
    綜合分析端點：支援圖片 + 文字
    """
    try:
        # 取得輸入資料
        image_file = request.files.get('image', None)
        text_input = request.form.get('text', '').strip()
        
        extracted_text = ''
        ocr_performed = False
        
        # 處理圖片 OCR（如果有提供）
        if image_file:
            logger.info("處理上傳的圖片...")
            extracted_text, error = perform_ocr(image_file)
            
            if error:
                # OCR 失敗但有文字輸入時，繼續處理文字
                if text_input:
                    logger.warning(f"OCR 失敗但有文字輸入，繼續處理: {error}")
                else:
                    return jsonify(error), 400
            else:
                ocr_performed = True
                
                # 檢查 OCR 結果
                if len(extracted_text) < MIN_TEXT_LENGTH and not text_input:
                    return jsonify({
                        'error': f'圖片辨識的文字太少，請重新上傳更清晰的圖片或補充文字說明',
                        'code': 'INSUFFICIENT_OCR_TEXT'
                    }), 400
        
        # 合併文字
        full_text = ' '.join(filter(None, [extracted_text, text_input])).strip()
        
        # 驗證合併後的文字
        if not full_text:
            return jsonify({
                'error': '請提供圖片或文字進行分析',
                'code': 'NO_CONTENT'
            }), 400
        
        if len(full_text) < MIN_TEXT_LENGTH:
            return jsonify({
                'error': f'總文字長度至少需要 {MIN_TEXT_LENGTH} 個字元',
                'code': 'COMBINED_TEXT_TOO_SHORT'
            }), 400
        
        # 執行分析
        logger.info(f"開始分析文字 (總長度: {len(full_text)} 字元)")
        analysis_result = detector_instance.analyze(full_text)
        
        # 添加額外資訊
        analysis_result['sources'] = {
            'has_image': bool(image_file),
            'has_text': bool(text_input),
            'ocr_performed': ocr_performed,
            'total_length': len(full_text)
        }
        
        if ocr_performed and extracted_text:
            # 只顯示部分 OCR 文字以保護隱私
            analysis_result['ocr_preview'] = (
                extracted_text[:100] + '...' 
                if len(extracted_text) > 100 
                else extracted_text
            )
        
        if 'error' in analysis_result:
            return jsonify(analysis_result), 500

        return jsonify(analysis_result), 200

    except Exception as e:
        logger.error(f"綜合分析時發生錯誤: {e}", exc_info=True)
        return jsonify({
            'error': '處理請求時發生錯誤',
            'details': str(e),
            'code': 'GENERAL_ERROR'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """取得模型資訊"""
    info = {
        "model_loaded": detector_instance.model_loaded,
        "model_path": detector_instance.model_path,
        "device": str(detector_instance.device),
        "ocr_enabled": bool(OCR_API_KEY),
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // 1024 // 1024,
        "min_text_length": MIN_TEXT_LENGTH,
        "max_text_length": MAX_TEXT_LENGTH
    }
    return jsonify(info), 200

if __name__ == '__main__':
    # 生產環境設定
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"🚀 啟動反詐騙 API 服務 (Port: {port}, Debug: {debug})")
    
    # 注意：生產環境應使用 WSGI 伺服器如 Gunicorn
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port,
        threaded=True  # 啟用多執行緒
    )