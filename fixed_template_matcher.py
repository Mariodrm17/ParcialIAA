import cv2
import numpy as np
import os
import json
from collections import Counter

class FixedTemplateRecognizer:
    """
    Reconocedor MEJORADO que usa las dimensiones correctas
    Compara esquinas Y palos por separado para mayor precisión
    """
    
    def __init__(self):
        self.templates_dir = "card_templates"
        self.corners_dir = os.path.join(self.templates_dir, "corners")
        self.suits_dir = os.path.join(self.templates_dir, "suits")
        self.config_file = "templates_config.json"
        
        # Plantillas cargadas
        self.corner_templates = {}  # Esquinas (número + palo)
        self.number_templates = {}  # Solo números
        self.suit_templates = {}    # Solo palos
        
        self.suits_names = {
            'H': 'Corazones',
            'D': 'Diamantes', 
            'C': 'Treboles',
            'S': 'Picas'
        }
        
        self.load_templates()
    
    def extract_corner(self, card_img):
        """
        Extrae esquina con las dimensiones CORRECTAS
        35% alto x 30% ancho (ajustado para capturar número + palo)
        """
        height, width = card_img.shape[:2]
        
        # MEJORADO: dimensiones más generosas
        corner_h = int(height * 0.35)
        corner_w = int(width * 0.30)
        
        corner = card_img[0:corner_h, 0:corner_w]
        
        # Redimensionar a tamaño estándar MÁS GRANDE
        standard_size = (100, 140)  # Aumentado de (80, 120)
        corner_resized = cv2.resize(corner, standard_size)
        
        return corner_resized
    
    def extract_number_region(self, card_img):
        """Extrae solo la región del número"""
        height, width = card_img.shape[:2]
        
        num_h = int(height * 0.25)
        num_w = int(width * 0.25)
        
        number = card_img[0:num_h, 0:num_w]
        
        # Tamaño estándar para números
        standard_size = (80, 80)
        number_resized = cv2.resize(number, standard_size)
        
        return number_resized
    
    def extract_suit_region(self, card_img):
        """Extrae solo la región del palo"""
        height, width = card_img.shape[:2]
        
        # Región del palo
        suit_y1 = int(height * 0.15)
        suit_y2 = int(height * 0.45)
        suit_x1 = int(width * 0.05)
        suit_x2 = int(width * 0.35)
        
        suit = card_img[suit_y1:suit_y2, suit_x1:suit_x2]
        
        # Tamaño estándar para palos
        standard_size = (60, 90)
        suit_resized = cv2.resize(suit, standard_size)
        
        return suit_resized
    
    def preprocess_region(self, region_img):
        """
        Preprocesa región para comparación
        """
        # Convertir a escala de grises
        if len(region_img.shape) == 3:
            gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = region_img
        
        # CLAHE para normalizar iluminación
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Reducir ruido
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        return binary
    
    def load_templates(self):
        """
        Carga plantillas desde el nuevo formato
        """
        if not os.path.exists(self.config_file):
            print("⚠️  No se encontró archivo de configuración")
            print("   Ejecuta: python template_capture.py")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            templates_loaded = 0
            corners_loaded = 0
            numbers_loaded = 0
            suits_loaded = 0
            
            for template_info in config.get('templates', []):
                card_key = template_info['card_name']
                files = template_info.get('files', {})
                
                # Cargar esquina completa
                if 'corner' in files:
                    corner_path = os.path.join(self.corners_dir, files['corner'])
                    if os.path.exists(corner_path):
                        corner_img = cv2.imread(corner_path)
                        if corner_img is not None:
                            corner_processed = self.preprocess_region(corner_img)
                            
                            if card_key not in self.corner_templates:
                                self.corner_templates[card_key] = []
                            
                            self.corner_templates[card_key].append({
                                'processed': corner_processed,
                                'color': corner_img,
                                'full_name': template_info['full_name'],
                                'number': template_info['number'],
                                'suit': template_info['suit']
                            })
                            corners_loaded += 1
                
                # Cargar número
                if 'number' in files:
                    number_path = os.path.join(self.corners_dir, files['number'])
                    if os.path.exists(number_path):
                        number_img = cv2.imread(number_path)
                        if number_img is not None:
                            number_processed = self.preprocess_region(number_img)
                            
                            number_key = template_info['number']
                            if number_key not in self.number_templates:
                                self.number_templates[number_key] = []
                            
                            self.number_templates[number_key].append({
                                'processed': number_processed,
                                'color': number_img
                            })
                            numbers_loaded += 1
                
                # Cargar palo
                if 'suit' in files:
                    suit_path = os.path.join(self.suits_dir, files['suit'])
                    if os.path.exists(suit_path):
                        suit_img = cv2.imread(suit_path)
                        if suit_img is not None:
                            suit_processed = self.preprocess_region(suit_img)
                            
                            suit_key = template_info['suit']
                            if suit_key not in self.suit_templates:
                                self.suit_templates[suit_key] = []
                            
                            self.suit_templates[suit_key].append({
                                'processed': suit_processed,
                                'color': suit_img,
                                'name': self.suits_names[suit_key]
                            })
                            suits_loaded += 1
                
                templates_loaded += 1
            
            print(f"✅ Plantillas cargadas:")
            print(f"   • Cartas totales: {templates_loaded}")
            print(f"   • Esquinas: {corners_loaded}")
            print(f"   • Números únicos: {len(self.number_templates)}")
            print(f"   • Palos únicos: {len(self.suit_templates)}")
            
            return templates_loaded > 0
            
        except Exception as e:
            print(f"❌ Error cargando plantillas: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compare_regions(self, region1, region2):
        """
        Compara dos regiones usando múltiples métodos
        Retorna score de 0 a 1
        """
        # Asegurar mismo tamaño
        if region1.shape != region2.shape:
            region2 = cv2.resize(region2, (region1.shape[1], region1.shape[0]))
        
        scores = []
        
        # Método 1: Template Matching (Correlación normalizada)
        try:
            result = cv2.matchTemplate(region1, region2, cv2.TM_CCOEFF_NORMED)
            score_tm = result[0][0]
            scores.append(max(0, score_tm))
        except:
            scores.append(0)
        
        # Método 2: Diferencia estructural
        try:
            r1_norm = cv2.normalize(region1, None, 0, 255, cv2.NORM_MINMAX)
            r2_norm = cv2.normalize(region2, None, 0, 255, cv2.NORM_MINMAX)
            
            diff = cv2.absdiff(r1_norm, r2_norm)
            score_diff = 1.0 - (np.mean(diff) / 255.0)
            scores.append(max(0, score_diff))
        except:
            scores.append(0)
        
        # Método 3: Histograma
        try:
            hist1 = cv2.calcHist([region1], [0], None, [32], [0, 256])
            hist2 = cv2.calcHist([region2], [0], None, [32], [0, 256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            score_hist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            scores.append(max(0, score_hist))
        except:
            scores.append(0)
        
        # Promedio ponderado (más peso al template matching)
        if len(scores) >= 3:
            final_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
        elif len(scores) > 0:
            final_score = np.mean(scores)
        else:
            final_score = 0.0
        
        return max(0, min(1, final_score))
    
    def recognize_number(self, card_img):
        """Reconoce solo el número"""
        if not self.number_templates:
            return None, 0.0
        
        number_region = self.extract_number_region(card_img)
        number_processed = self.preprocess_region(number_region)
        
        best_number = None
        best_score = 0.0
        
        for number_key, templates_list in self.number_templates.items():
            scores_for_number = []
            
            for template_data in templates_list:
                template_processed = template_data['processed']
                score = self.compare_regions(number_processed, template_processed)
                scores_for_number.append(score)
            
            max_score = max(scores_for_number) if scores_for_number else 0.0
            
            if max_score > best_score:
                best_score = max_score
                best_number = number_key
        
        return best_number, best_score
    
    def recognize_suit(self, card_img):
        """Reconoce solo el palo"""
        if not self.suit_templates:
            return None, 0.0
        
        suit_region = self.extract_suit_region(card_img)
        suit_processed = self.preprocess_region(suit_region)
        
        best_suit = None
        best_score = 0.0
        
        for suit_key, templates_list in self.suit_templates.items():
            scores_for_suit = []
            
            for template_data in templates_list:
                template_processed = template_data['processed']
                score = self.compare_regions(suit_processed, template_processed)
                scores_for_suit.append(score)
            
            max_score = max(scores_for_suit) if scores_for_suit else 0.0
            
            if max_score > best_score:
                best_score = max_score
                best_suit = suit_key
        
        return best_suit, best_score
    
    def recognize_card(self, card_img):
        """
        Reconoce carta COMPLETA usando método HÍBRIDO
        Combina reconocimiento de esquina + número + palo
        """
        if not self.corner_templates:
            print("⚠️  No hay plantillas cargadas")
            return None, 0.0
        
        try:
            # Método 1: Reconocimiento por esquina completa
            corner = self.extract_corner(card_img)
            corner_processed = self.preprocess_region(corner)
            
            corner_results = []
            
            for card_key, templates_list in self.corner_templates.items():
                scores_for_card = []
                
                for template_data in templates_list:
                    template_corner = template_data['processed']
                    score = self.compare_regions(corner_processed, template_corner)
                    scores_for_card.append(score)
                
                max_score = max(scores_for_card) if scores_for_card else 0.0
                corner_results.append((templates_list[0]['full_name'], max_score, card_key))
            
            corner_results.sort(key=lambda x: x[1], reverse=True)
            
            # Método 2: Reconocimiento por número + palo por separado
            number, num_conf = self.recognize_number(card_img)
            suit, suit_conf = self.recognize_suit(card_img)
            
            # Combinar resultados
            if number and suit:
                # Buscar carta combinada
                combined_name = f"{number} de {self.suits_names.get(suit, '?')}"
                combined_conf = (num_conf + suit_conf) / 2.0
                
                # Si la combinación coincide con el mejor de esquina
                if corner_results and corner_results[0][0] == combined_name:
                    # Promedio ponderado (más peso a esquina que es más específica)
                    final_conf = (corner_results[0][1] * 0.6 + combined_conf * 0.4)
                    return combined_name, final_conf
                elif combined_conf >= 0.5:
                    # Usar combinación si es suficientemente buena
                    return combined_name, combined_conf
            
            # Fallback: usar mejor de esquina
            if corner_results and corner_results[0][1] >= 0.5:
                return corner_results[0][0], corner_results[0][1]
            
            # No hay match suficientemente bueno
            if corner_results:
                return None, corner_results[0][1]
            else:
                return None, 0.0
        
        except Exception as e:
            print(f"❌ Error reconociendo: {e}")
            return None, 0.0
    
    def recognize_with_rotation(self, card_img):
        """
        Intenta reconocer con pequeñas rotaciones
        """
        results = []
        
        # Sin rotación
        name, conf = self.recognize_card(card_img)
        if name:
            results.append((name, conf))
        
        # Rotaciones pequeñas: -8, -4, +4, +8 grados
        for angle in [-8, -4, 4, 8]:
            height, width = card_img.shape[:2]
            center = (width // 2, height // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(card_img, M, (width, height), 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            name, conf = self.recognize_card(rotated)
            if name:
                results.append((name, conf * 0.95))  # Penalizar ligeramente rotaciones
        
        # Retornar el mejor resultado
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            return results[0]
        
        return None, 0.0
    
    def save_debug_image(self, card_img, detected_name, confidence):
        """Guarda imagen para debug"""
        try:
            corner = self.extract_corner(card_img)
            number = self.extract_number_region(card_img)
            suit = self.extract_suit_region(card_img)
            
            corner_processed = self.preprocess_region(corner)
            number_processed = self.preprocess_region(number)
            suit_processed = self.preprocess_region(suit)
            
            # Crear imagen de debug con todas las regiones
            row1 = np.hstack([
                cv2.resize(card_img, (200, 300)),
                cv2.resize(cv2.cvtColor(corner_processed, cv2.COLOR_GRAY2BGR), (100, 150))
            ])
            
            row2 = np.hstack([
                cv2.resize(cv2.cvtColor(number_processed, cv2.COLOR_GRAY2BGR), (100, 100)),
                cv2.resize(cv2.cvtColor(suit_processed, cv2.COLOR_GRAY2BGR), (100, 150))
            ])
            
            # Agregar padding
            row2_padded = cv2.copyMakeBorder(row2, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            debug_img = np.vstack([row1, row2_padded])
            
            # Agregar texto
            cv2.putText(debug_img, f"{detected_name if detected_name else 'NO MATCH'}", 
                       (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_img, f"Conf: {confidence:.1%}", 
                       (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"debug_{timestamp}_{confidence:.0%}.jpg"
            cv2.imwrite(filename, debug_img)
            print(f"💾 Debug guardado: {filename}")
        except Exception as e:
            print(f"Error guardando debug: {e}")


def test_recognition_live():
    """Prueba el reconocedor en tiempo real"""
    
    print("=" * 70)
    print("🎯 RECONOCEDOR MEJORADO - ESQUINAS Y PALOS")
    print("=" * 70)
    print("\n✨ MEJORAS:")
    print("✓ Dimensiones correctas (35% x 30%)")
    print("✓ Reconocimiento híbrido (esquina + número + palo)")
    print("✓ Mayor precisión en comparaciones")
    print("✓ Tolerancia a rotaciones mejorada")
    print("=" * 70)
    
    recognizer = FixedTemplateRecognizer()
    
    if not recognizer.corner_templates:
        print("\n❌ No hay plantillas. Ejecuta: python template_capture.py")
        return
    
    # Cargar calibración
    if not os.path.exists("green_calibration.json"):
        print("\n❌ No hay calibración. Ejecuta: python green_calibrator.py")
        return
    
    with open("green_calibration.json", 'r') as f:
        calibration = json.load(f)
    
    green_lower = np.array(calibration['green_lower'])
    green_upper = np.array(calibration['green_upper'])
    
    camera_idx = input("\nÍndice de cámara (default=0): ").strip()
    camera_idx = int(camera_idx) if camera_idx else 0
    
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara {camera_idx}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    print("\n✅ Sistema iniciado")
    print("Teclas: 'q'=Salir | 'r'=Reset | 'd'=Debug")
    
    detected_cards = set()
    debug_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Detectar cartas
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        cards_mask = cv2.bitwise_not(green_mask)
        contours, _ = cv2.findContours(cards_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        cards_found = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 2000 < area < 50000:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.5 <= aspect_ratio <= 0.9:
                        # Extraer carta
                        margin = 20
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(frame.shape[1], x + w + margin)
                        y2 = min(frame.shape[0], y + h + margin)
                        
                        card_img = frame[y1:y2, x1:x2]
                        
                        if card_img.size > 0:
                            cards_found += 1
                            
                            # Reconocer con rotaciones
                            card_name, confidence = recognizer.recognize_with_rotation(card_img)
                            
                            # Color según confianza
                            if confidence >= 0.7:
                                color = (0, 255, 0)  # Verde
                            elif confidence >= 0.5:
                                color = (0, 255, 255)  # Amarillo
                            else:
                                color = (0, 0, 255)  # Rojo
                            
                            # Dibujar
                            cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
                            
                            if card_name:
                                cv2.putText(display, card_name, (x, y-30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                cv2.putText(display, f"{confidence:.1%}", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                if confidence >= 0.6:
                                    detected_cards.add(card_name)
                                
                                if debug_mode:
                                    recognizer.save_debug_image(card_img, card_name, confidence)
                                    debug_mode = False
                            else:
                                cv2.putText(display, f"? ({confidence:.1%})", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info
        cv2.putText(display, f"Cartas: {cards_found} | Reconocidas: {len(detected_cards)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "q=Salir | r=Reset | d=Debug", 
                   (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Reconocedor Mejorado", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            detected_cards.clear()
            print("🧹 Lista limpiada")
        elif key == ord('d'):
            debug_mode = True
            print("📸 Debug activado")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Resultados
    print("\n" + "=" * 70)
    print("📊 CARTAS DETECTADAS")
    print("=" * 70)
    
    if detected_cards:
        for i, card in enumerate(sorted(detected_cards), 1):
            print(f"{i:2d}. {card}")
    else:
        print("❌ No se detectaron cartas")
    
    print("=" * 70)


if __name__ == "__main__":
    from datetime import datetime
    test_recognition_live()