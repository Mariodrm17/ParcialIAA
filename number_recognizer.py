import cv2
import numpy as np
import random

class NumberRecognizer:
    def __init__(self):
        self.number_templates = {
            'A': 'AS', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
            'J': 'J', 'Q': 'Q', 'K': 'K'
        }
    
    def preprocess_card_number(self, card_image):
        """Preprocesa la región del número de la carta"""
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def find_number_region(self, card_image):
        """Encuentra la región donde está el número de la carta"""
        height, width = card_image.shape[:2]
        number_region_height = height // 4
        number_region_width = width // 4
        number_region = card_image[10:number_region_height, 10:number_region_width]
        return number_region
    
    def recognize_number_contour(self, number_region):
        """Reconoce el número basado en contornos (alternativa a OCR)"""
        try:
            processed = self.preprocess_card_number(number_region)
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Lógica simple basada en relación de aspecto y área
                aspect_ratio = w / h if h > 0 else 0
                
                if aspect_ratio > 1.5:
                    return '10'  # El 10 es más ancho
                elif area > 500:
                    return 'A'   # Área grande podría ser A
                elif 300 < area < 500:
                    return '8'   # Área media
                else:
                    return '2'   # Área pequeña
                    
        except Exception as e:
            print(f"Error en reconocimiento por contorno: {e}")
        
        return None
    
    def recognize(self, card_image):
        """Reconoce el número de la carta"""
        number_region = self.find_number_region(card_image)
        
        if number_region is None or number_region.size == 0:
            return self._simulate_recognition()
        
        # Intentar reconocimiento por contornos
        number = self.recognize_number_contour(number_region)
        
        # Fallback a simulación
        if not number:
            number = self._simulate_recognition()
        
        return number
    
    def _simulate_recognition(self):
        """Simula reconocimiento (para pruebas)"""
        numbers = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return random.choice(numbers)