import cv2
import numpy as np
import os
import json

class CardTemplateRecognizer:
    """Reconocedor PRECISO de cartas usando plantillas"""
    
    def __init__(self):
        self.templates_dir = "card_templates"
        self.config_file = "templates_config.json"
        self.templates = {}
        self.suits_names = {
            'H': 'Corazones',
            'D': 'Diamantes', 
            'C': 'Treboles',
            'S': 'Picas'
        }
        
        # Cargar plantillas
        self.load_templates()
    
    def load_templates(self):
        """Carga todas las plantillas disponibles"""
        if not os.path.exists(self.config_file):
            print("⚠️  No se encontró archivo de configuración de plantillas")
            print("   Ejecuta primero: python template_capture.py")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            templates_loaded = 0
            
            for template_info in config['templates']:
                filepath = os.path.join(self.templates_dir, template_info['filename'])
                
                if os.path.exists(filepath):
                    # Cargar imagen
                    img = cv2.imread(filepath)
                    
                    if img is not None:
                        # Preprocesar plantilla
                        processed = self.preprocess_card(img)
                        
                        card_key = template_info['card_name']
                        self.templates[card_key] = {
                            'image': processed,
                            'original': img,
                            'full_name': template_info['full_name'],
                            'number': template_info['number'],
                            'suit': template_info['suit']
                        }
                        templates_loaded += 1
            
            print(f"✅ Plantillas cargadas: {templates_loaded}")
            return templates_loaded > 0
            
        except Exception as e:
            print(f"❌ Error cargando plantillas: {e}")
            return False
    
    def preprocess_card(self, card_img):
        """Preprocesa carta para comparación"""
        try:
            # Redimensionar a tamaño estándar
            resized = cv2.resize(card_img, (200, 300))
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Reducir ruido
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            return denoised
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return card_img
    
    def recognize_card(self, card_img, method='correlation'):
        """
        Reconoce una carta comparándola con plantillas
        
        Métodos disponibles:
        - 'correlation': Correlación cruzada normalizada (RÁPIDO)
        - 'orb': ORB feature matching (MÁS PRECISO pero más lento)
        - 'sift': SIFT feature matching (MUY PRECISO pero requiere opencv-contrib)
        """
        if not self.templates:
            print("⚠️  No hay plantillas cargadas")
            return None, 0.0
        
        # Preprocesar carta a reconocer
        processed_card = self.preprocess_card(card_img)
        
        if method == 'correlation':
            return self._recognize_by_correlation(processed_card)
        elif method == 'orb':
            return self._recognize_by_orb(card_img)
        else:
            return self._recognize_by_correlation(processed_card)
    
    def _recognize_by_correlation(self, card_gray):
        """Reconocimiento por correlación (RÁPIDO Y EFECTIVO)"""
        best_match = None
        best_score = 0.0
        
        for card_key, template_data in self.templates.items():
            template_gray = template_data['image']
            
            # Asegurar mismo tamaño
            if card_gray.shape != template_gray.shape:
                card_gray = cv2.resize(card_gray, template_gray.shape[::-1])
            
            # Calcular correlación
            correlation = cv2.matchTemplate(card_gray, template_gray, 
                                           cv2.TM_CCOEFF_NORMED)
            score = correlation[0][0]
            
            if score > best_score:
                best_score = score
                best_match = template_data['full_name']
        
        # Umbral de confianza
        confidence_threshold = 0.6
        
        if best_score >= confidence_threshold:
            return best_match, best_score
        else:
            return None, best_score
    
    def _recognize_by_orb(self, card_img):
        """Reconocimiento por ORB features (MÁS ROBUSTO)"""
        
        # Inicializar ORB
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Detectar keypoints y descriptores de la carta
        gray_card = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        kp_card, des_card = orb.detectAndCompute(gray_card, None)
        
        if des_card is None:
            return None, 0.0
        
        best_match = None
        best_score = 0
        
        # Comparar con cada plantilla
        for card_key, template_data in self.templates.items():
            template_img = template_data['original']
            gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            
            # Detectar keypoints de la plantilla
            kp_template, des_template = orb.detectAndCompute(gray_template, None)
            
            if des_template is None:
                continue
            
            # Hacer matching con BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_card, des_template)
            
            # Calcular score basado en número de matches
            score = len(matches)
            
            if score > best_score:
                best_score = score
                best_match = template_data['full_name']
        
        # Normalizar score (asumiendo máximo de 100 matches buenos)
        normalized_score = min(best_score / 50.0, 1.0)
        
        # Umbral mínimo
        if best_score >= 15:  # Al menos 15 matches
            return best_match, normalized_score
        else:
            return None, normalized_score
    
    def recognize_multiple(self, card_images, method='correlation'):
        """Reconoce múltiples cartas"""
        results = []
        
        for i, card_img in enumerate(card_images):
            card_name, confidence = self.recognize_card(card_img, method)
            
            results.append({
                'index': i,
                'card': card_name,
                'confidence': confidence,
                'image': card_img
            })
        
        return results
    
    def test_recognition_live(self, camera_index=0, method='correlation'):
        """Prueba el reconocimiento en tiempo real"""
        
        if not self.templates:
            print("❌ No hay plantillas cargadas")
            return
        
        # Cargar calibración del tapete
        if not os.path.exists("green_calibration.json"):
            print("❌ No se encontró calibración del tapete verde")
            return
        
        with open("green_calibration.json", 'r') as f:
            calibration = json.load(f)
        
        green_lower = np.array(calibration['green_lower'])
        green_upper = np.array(calibration['green_upper'])
        
        print(f"\n🧪 PRUEBA DE RECONOCIMIENTO - Método: {method}")
        print("Presiona 'q' para salir")
        print("Presiona 'm' para cambiar método")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        current_method = method
        
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
            
            detected_cards = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 2000 < area < 50000:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        if 0.5 <= aspect_ratio <= 0.9:
                            margin = 15
                            x1 = max(0, x - margin)
                            y1 = max(0, y - margin)
                            x2 = min(frame.shape[1], x + w + margin)
                            y2 = min(frame.shape[0], y + h + margin)
                            
                            card_img = frame[y1:y2, x1:x2]
                            
                            if card_img.size > 0:
                                # Reconocer carta
                                card_name, confidence = self.recognize_card(
                                    card_img, method=current_method
                                )
                                
                                detected_cards.append({
                                    'position': (x, y, w, h),
                                    'name': card_name,
                                    'confidence': confidence
                                })
            
            # Dibujar resultados
            for card_data in detected_cards:
                x, y, w, h = card_data['position']
                card_name = card_data['name']
                confidence = card_data['confidence']
                
                # Color según confianza
                if confidence >= 0.8:
                    color = (0, 255, 0)  # Verde
                elif confidence >= 0.6:
                    color = (0, 255, 255)  # Amarillo
                else:
                    color = (0, 0, 255)  # Rojo
                
                # Dibujar rectángulo
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
                
                # Dibujar nombre
                if card_name:
                    text = f"{card_name} ({confidence:.2f})"
                    cv2.putText(display, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    text = f"? ({confidence:.2f})"
                    cv2.putText(display, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Información
            cv2.putText(display, f"Metodo: {current_method} | Cartas: {len(detected_cards)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "q=Salir | m=Cambiar metodo", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Reconocimiento de Cartas", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                current_method = 'orb' if current_method == 'correlation' else 'correlation'
                print(f"🔄 Método cambiado a: {current_method}")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Menú principal"""
    recognizer = CardTemplateRecognizer()
    
    if not recognizer.templates:
        print("\n⚠️  No hay plantillas disponibles")
        print("   Ejecuta primero: python template_capture.py")
        print("   Para capturar plantillas de las cartas")
        return
    
    while True:
        print("\n" + "=" * 60)
        print("🎯 RECONOCEDOR DE CARTAS POR PLANTILLAS")
        print("=" * 60)
        print(f"📊 Plantillas disponibles: {len(recognizer.templates)}")
        print("=" * 60)
        print("1. 🧪 Probar reconocimiento en vivo (Correlación - RÁPIDO)")
        print("2. 🎯 Probar reconocimiento en vivo (ORB - PRECISO)")
        print("3. 📋 Listar plantillas disponibles")
        print("4. ❌ Salir")
        print("=" * 60)
        
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            camera = input("Índice de cámara (default=0): ").strip()
            camera_idx = int(camera) if camera else 0
            recognizer.test_recognition_live(camera_idx, method='correlation')
        
        elif choice == '2':
            camera = input("Índice de cámara (default=0): ").strip()
            camera_idx = int(camera) if camera else 0
            recognizer.test_recognition_live(camera_idx, method='orb')
        
        elif choice == '3':
            print("\n📋 PLANTILLAS DISPONIBLES:")
            for card_key, data in recognizer.templates.items():
                print(f"  ✅ {data['full_name']}")
        
        elif choice == '4':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción no válida")


if __name__ == "__main__":
    main()