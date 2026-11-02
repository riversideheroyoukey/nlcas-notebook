# ui/apfel_client.py
# Version: v0.9.6
# Last Updated: 2026-04-18
#
# Description:
#   Apple Vision Framework (Apfel) bridge for macOS.
#   Provides near-instant text and symbol recognition using Apple Silicon NPU.

import Vision
import Quartz
from Cocoa import NSData
import base64

def run_apfel_ocr(image_b64):
    """Base64画像を受け取り、Apple Visionでテキスト認識を実行する"""
    try:
        # Base64デコード
        image_data = base64.b64decode(image_b64)
        ns_data = NSData.dataWithBytes_length_(image_data, len(image_data))
        
        # リクエストの作成
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        
        # ハンドラ作成と実行
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        success, error = handler.performRequests_error_([request], None)
        
        if not success:
            return {"ok": False, "error": str(error)}

        # 結果の抽出（最も確信度の高い候補を連結）
        results = request.results()
        text_list = []
        confidences = []
        
        for observation in results:
            candidate = observation.topCandidates_(1)[0]
            text_list.append(candidate.string())
            confidences.append(candidate.confidence())

        # 暫定的なLaTeX整形（Apfelは素のテキストを返すため）
        recognized_text = " ".join(text_list)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "ok": True, 
            "latex": recognized_text, 
            "confidence": avg_confidence,
            "engine": "apfel"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}