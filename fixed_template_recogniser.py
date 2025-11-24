import cv2
import numpy as np
import os
import json
from collections import Counter

class FixedTemplateRecognizer:
    """
    Template Matcher ARREGLADO
    Compara SOLO las esquinas, no la carta completa
    """
    
    def __init__(self):
        self.templates_dir = "card_templates"
        self.config_file = "templates_config.json"
        self.corner_templates = {}  # Solo esquinas
        self.suits_names = {
            'H': 'Corazones',
            'D': 'Diamantes', 
            'C': 'Treboles',
            'S': 'Picas'
        }
        
        self.load_templates()
    
    def extract_corner(self, card_img):
        """
        Extrae la esquina superior izquierda donde está el valor
        Esta es la región que vamos a comparar
        """
        height, width = card_img.shape[:2]
        
        # Extraer aproximadamente 25% ancho x 30% alto
        corner_h = int(height * 0.30)
        corner_w = int(width * 0.25)
        
        corner = card_img[0:corner_h, 0:corner_w]
        
        # Redimensionar a tamaño estándar
        standard_size = (80, 120)  # Tamaño estándar de esquina
        corner_resized = cv2.resize(corner, standard_size)
        
        return corner_resized
    
    def preprocess_corner(self, corner_img):
        """
        Preprocesa la esquina para comparación
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
        
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
        Carga plantillas y extrae SOLO las esquinas
        """
        if not os.path.exists(self.config_file):
            print("⚠️  No se encontró archivo de configuración")
            print("   Ejecuta: python template_capture.py")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            templates_loaded = 0
            
            for template_info in config['templates']:
                filepath = os.path.join(self.templates_dir, template_info['filename'])
                
                if os.path.exists(filepath):
                    # Cargar imagen completa
                    img = cv2.imread(filepath)
                    
                    if img is not None:
                        # Extraer SOLO la esquina
                        corner = self.extract_corner(img)
                        corner_processed = self.preprocess_corner(corner)
                        
                        card_key = template_info['card_name']
                        
                        # Guardar esquina procesada
                        if card_key not in self.corner_templates:
                            self.corner_templates[card_key] = []
                        
                        self.corner_templates[card_key].append({
                            'corner': corner_processed,
                            'corner_color': corner,  # También en color
                            'full_name': template_info['full_name'],
                            'number': template_info['number'],
                            'suit': template_info['suit']
                        })
                        
                        templates_loaded += 1
            
            print(f"✅ Plantillas de esquinas cargadas: {templates_loaded}")
            print(f"🎴 Cartas únicas: {len(self.corner_templates)}")
            
            return templates_loaded > 0
            
        except Exception as e:
            print(f"❌ Error cargando plantillas: {e}")
            return False
    
    def compare_corners(self, corner1, corner2):
        """
        Compara dos esquinas usando múltiples métodos
        Retorna score de 0 a 1
        """
        # Asegurar mismo tamaño
        if corner1.shape != corner2.shape:
            corner2 = cv2.resize(corner2, (corner1.shape[1], corner1.shape[0]))
        
        scores = []
        
        # Método 1: Template Matching (Correlación)
        result = cv2.matchTemplate(corner1, corner2, cv2.TM_CCOEFF_NORMED)
        score_tm = result[0][0]
        scores.append(score_tm)
        
        # Método 2: Structural Similarity (SSIM) - más robusto
        # Normalizar
        c1_norm = cv2.normalize(corner1, None, 0, 255, cv2.NORM_MINMAX)
        c2_norm = cv2.normalize(corner2, None, 0, 255, cv2.NORM_MINMAX)
        
        # Calcular diferencia absoluta
        diff = cv2.absdiff(c1_norm, c2_norm)
        score_diff = 1.0 - (np.mean(diff) / 255.0)
        scores.append(score_diff)
        
        # Método 3: Histograma
        hist1 = cv2.calcHist([corner1], [0], None, [32], [0, 256])
        hist2 = cv2.calcHist([corner2], [0], None, [32], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        score_hist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        scores.append(score_hist)
        
        # Promedio ponderado
        # TM es el más importante
        final_score = (score_tm * 0.5 + score_diff * 0.3 + score_hist * 0.2)
        
        return max(0, min(1, final_score))
    
    def recognize_card(self, card_img):
        """
        Reconoce carta comparando SOLO la esquina
        Retorna (nombre, confianza)
        """
        if not self.corner_templates:
            print("⚠️  No hay plantillas cargadas")
            return None, 0.0
        
        try:
            # Extraer esquina de la carta a reconocer
            corner = self.extract_corner(card_img)
            corner_processed = self.preprocess_corner(corner)
            
            best_match = None
            best_score = 0.0
            all_scores = []
            
            # Comparar con todas las plantillas
            for card_key, templates_list in self.corner_templates.items():
                # Comparar con todas las versiones de esta carta
                scores_for_this_card = []
                
                for template_data in templates_list:
                    template_corner = template_data['corner']
                    
                    # Comparar esquinas
                    score = self.compare_corners(corner_processed, template_corner)
                    scores_for_this_card.append(score)
                
                # Usar el mejor score de todas las versiones
                max_score = max(scores_for_this_card) if scores_for_this_card else 0.0
                
                all_scores.append((templates_list[0]['full_name'], max_score))
                
                if max_score > best_score:
                    best_score = max_score
                    best_match = templates_list[0]['full_name']
            
            # Debug: mostrar top 3
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Umbral de confianza
            confidence_threshold = 0.5
            
            if best_score >= confidence_threshold:
                return best_match, best_score
            else:
                # Imprimir top 3 para debug
                print(f"\n🔍 Top 3 matches:")
                for i, (name, score) in enumerate(all_scores[:3], 1):
                    print(f"   {i}. {name}: {score:.3f}")
                return None, best_score
        
        except Exception as e:
            print(f"❌ Error reconociendo: {e}")
            return None, 0.0
    
    def recognize_with_rotation(self, card_img):
        """
        Intenta reconocer con pequeñas rotaciones
        Por si la carta está ligeramente inclinada
        """
        results = []
        
        # Sin rotación
        name, conf = self.recognize_card(card_img)
        if name:
            results.append((name, conf))
        
        # Rotaciones pequeñas: -5, +5 grados
        for angle in [-5, 5]:
            height, width = card_img.shape[:2]
            center = (width // 2, height // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(card_img, M, (width, height), 
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            name, conf = self.recognize_card(rotated)
            if name:
                results.append((name, conf))
        
        # Retornar el mejor resultado
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            return results[0]
        
        return None, 0.0
    
    def save_debug_image(self, card_img, detected_name, confidence):
        """Guarda imagen para debug"""
        corner = self.extract_corner(card_img)
        corner_processed = self.preprocess_corner(corner)
        
        # Crear imagen de debug
        debug_img = np.hstack([
            cv2.cvtColor(corner_processed, cv2.COLOR_GRAY2BGR),
            corner
        ])
        
        filename = f"debug_{detected_name}_{confidence:.2f}.jpg"
        cv2.imwrite(filename, debug_img)
        print(f"💾 Debug guardado: {filename}")


def test_recognition_live():
    """Prueba el reconocedor en tiempo real"""
    
    print("=" * 70)
    print("🎯 TEMPLATE MATCHER ARREGLADO")
    print("=" * 70)
    print("\n✨ MEJORAS:")
    print("✓ Compara SOLO las esquinas (no la carta completa)")
    print("✓ Múltiples métodos de comparación")
    print("✓ Normalización de iluminación mejorada")
    print("✓ Tolerancia a pequeñas rotaciones")
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
    print("Presiona: 'q'=Salir | 'r'=Resetear | 'd'=Guardar debug")
    
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
                            
                            # Reconocer (con rotaciones)
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
                                # Mostrar nombre y confianza
                                cv2.putText(display, card_name, (x, y-30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                cv2.putText(display, f"{confidence:.1%}", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Agregar a detectadas
                                if confidence >= 0.6:
                                    detected_cards.add(card_name)
                                
                                # Guardar debug si está activado
                                if debug_mode:
                                    recognizer.save_debug_image(card_img, card_name, confidence)
                                    debug_mode = False
                            else:
                                cv2.putText(display, f"? ({confidence:.1%})", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Info
        cv2.putText(display, f"Cartas visibles: {cards_found}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Reconocidas: {len(detected_cards)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "q=Salir | r=Reset | d=Debug", 
                   (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Template Matcher Arreglado", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            detected_cards.clear()
            print("🧹 Lista limpiada")
        elif key == ord('d'):
            debug_mode = True
            print("📸 Debug activado para siguiente carta")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Resultados
    print("\n" + "=" * 70)
    print("📊 CARTAS DETECTADAS")
    print("=" * 70)
    
    if detected_cards:
        for i, card in enumerate(sorted(detected_cards), 1):
            print(f"{i}. {card}")
    else:
        print("❌ No se detectaron cartas")
    
    print("=" * 70)


if __name__ == "__main__":
    test_recognition_live()