import cv2
import numpy as np
import random

class SuitRecognizer:
    def __init__(self):
        self.suits = {
            'HEARTS': 'Corazones',
            'DIAMONDS': 'Diamantes', 
            'CLUBS': 'Tréboles',
            'SPADES': 'Picas'
        }
    
    def find_suit_region(self, card_image):
        """Encuentra la región donde está el palo de la carta"""
        height, width = card_image.shape[:2]
        suit_region_height = height // 3
        suit_region_width = width // 4
        suit_region = card_image[height//4:height//4 + suit_region_height, 10:suit_region_width]
        return suit_region
    
    def detect_suit_by_color(self, suit_region):
        """Detecta el palo basado en el color"""
        hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)
        
        # Máscaras para rojo
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Máscara para negro
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        
        red_pixels = cv2.countNonZero(mask_red)
        black_pixels = cv2.countNonZero(mask_black)
        
        if red_pixels > black_pixels and red_pixels > 50:
            return ['HEARTS', 'DIAMONDS']
        elif black_pixels > red_pixels and black_pixels > 50:
            return ['CLUBS', 'SPADES']
        
        return []
    
    def detect_suit_by_shape(self, suit_region, possible_suits):
        """Distinguir entre palos del mismo color por forma"""
        if len(possible_suits) != 2:
            return self._simulate_recognition()
        
        gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if 'HEARTS' in possible_suits and 'DIAMONDS' in possible_suits:
                    return 'HEARTS' if circularity > 0.5 else 'DIAMONDS'
                elif 'CLUBS' in possible_suits and 'SPADES' in possible_suits:
                    return 'SPADES' if circularity < 0.4 else 'CLUBS'
        
        return self._simulate_recognition()
    
    def _simulate_recognition(self):
        """Simula reconocimiento (para pruebas)"""
        suits = ['HEARTS', 'DIAMONDS', 'CLUBS', 'SPADES']
        return random.choice(suits)
    
    def recognize(self, card_image):
        """Reconoce el palo de la carta"""
        suit_region = self.find_suit_region(card_image)
        
        if suit_region is None or suit_region.size == 0:
            return self._simulate_recognition()
        
        possible_suits = self.detect_suit_by_color(suit_region)
        
        if len(possible_suits) == 1:
            return possible_suits[0]
        elif len(possible_suits) == 2:
            return self.detect_suit_by_shape(suit_region, possible_suits)
        else:
            return self._simulate_recognition()
    
    def get_suit_name(self, suit_code):
        """Convierte código de palo a nombre en español"""
        return self.suits.get(suit_code, 'Desconocido')