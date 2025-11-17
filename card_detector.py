import cv2
import numpy as np

class IVCardDetector:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.min_card_area = 2000  # Más permisivo
        self.max_card_area = 100000 # Más grande
        self.card_width = 200
        self.card_height = 300
        
        # Rangos HSV por defecto - se ajustarán con calibración
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
        
    def detect_cards(self, frame):
        """Detección principal de cartas - VERSIÓN ROBUSTA"""
        try:
            if frame is None or frame.size == 0:
                print("❌ Frame vacío o None")
                return [], []
                
            original_frame = frame.copy()
            
            # Estrategia 1: Por color del tapete verde
            cards, positions = self._detect_by_green_background(original_frame)
            
            # Estrategia 2: Si no encuentra, por bordes
            if len(cards) == 0:
                if self.debug_mode:
                    print("⚠️  No se detectaron cartas por color, intentando por bordes...")
                cards, positions = self._detect_by_edges(original_frame)
            
            if self.debug_mode:
                print(f"🔍 Cartas detectadas: {len(cards)}")
                self._draw_detection_results(original_frame, positions)
            
            return cards, positions
            
        except Exception as e:
            print(f"❌ Error crítico en detección: {e}")
            return [], []
    
    def _detect_by_green_background(self, frame):
        """Detección por color del tapete verde"""
        try:
            # Preprocesamiento
            processed, scale = self._preprocess_frame(frame)
            
            # Crear máscara verde
            green_mask = self._create_green_mask(processed)
            
            if self.debug_mode:
                self._show_debug_frame("1 - Máscara Verde", green_mask)
            
            # Encontrar contornos
            contours = self._find_contours(green_mask)
            
            if self.debug_mode and len(contours) > 0:
                contour_frame = processed.copy()
                cv2.drawContours(contour_frame, contours, -1, (0, 0, 255), 2)
                self._show_debug_frame("2 - Contornos", contour_frame)
            
            # Filtrar y extraer cartas
            return self._filter_and_extract_cards(frame, contours, scale)
            
        except Exception as e:
            print(f"Error en detección por color: {e}")
            return [], []
    
    def _detect_by_edges(self, frame):
        """Detección alternativa por bordes"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            if self.debug_mode:
                self._show_debug_frame("3 - Bordes Canny", edges)
            
            # Operaciones morfológicas
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos rectangulares
            card_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_card_area < area < self.max_card_area:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Aceptar contornos con 4 lados
                    if len(approx) == 4:
                        card_contours.append(approx)
            
            return self._filter_and_extract_cards(frame, card_contours, 1.0)
            
        except Exception as e:
            print(f"Error en detección por bordes: {e}")
            return [], []
    
    def _preprocess_frame(self, frame):
        """Preprocesamiento del frame"""
        try:
            height, width = frame.shape[:2]
            scale = 1.0
            
            # Redimensionar si es muy grande
            if height > 800:
                scale = 800 / height
                new_width = int(width * scale)
                frame_resized = cv2.resize(frame, (new_width, 800))
            else:
                frame_resized = frame
            
            # Mejorar contraste
            lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced, scale
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return frame, 1.0
    
    def _create_green_mask(self, frame):
        """Crea máscara para el color verde"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Operaciones morfológicas
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            print(f"Error creando máscara verde: {e}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    def _find_contours(self, mask):
        """Encuentra contornos en la máscara"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except:
            return []
    
    def _filter_and_extract_cards(self, original_frame, contours, scale):
        """Filtra contornos y extrae imágenes de cartas"""
        card_images = []
        card_positions = []
        
        for contour in contours:
            try:
                area = cv2.contourArea(contour)
                if area < self.min_card_area or area > self.max_card_area:
                    continue
                
                # Escalar contorno si fue redimensionado
                if scale != 1.0:
                    contour_original = (contour / scale).astype(np.int32)
                else:
                    contour_original = contour
                
                # Aproximar a polígono
                epsilon = 0.02 * cv2.arcLength(contour_original, True)
                approx = cv2.approxPolyDP(contour_original, epsilon, True)
                
                # Buscar formas con 4 lados
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour_original)
                    
                    # Verificar relación de aspecto de carta
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.4 <= aspect_ratio <= 0.9:  # Más permisivo
                        # Extraer con margen
                        margin = 10
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(original_frame.shape[1], x + w + margin)
                        y2 = min(original_frame.shape[0], y + h + margin)
                        
                        card_img = original_frame[y1:y2, x1:x2]
                        
                        if card_img.size > 0 and card_img.shape[0] > 50 and card_img.shape[1] > 50:
                            card_images.append(card_img)
                            card_positions.append((x1, y1, x2-x1, y2-y1))
                            
            except Exception as e:
                if self.debug_mode:
                    print(f"Error procesando contorno: {e}")
                continue
        
        return card_images, card_positions
    
    def _show_debug_frame(self, title, frame):
        """Muestra frame para debug"""
        if self.debug_mode:
            try:
                # Redimensionar si es muy grande
                h, w = frame.shape[:2]
                if w > 600:
                    scale = 600 / w
                    new_h = int(h * scale)
                    display_frame = cv2.resize(frame, (600, new_h))
                else:
                    display_frame = frame
                
                cv2.imshow(title, display_frame)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Error mostrando debug: {e}")
    
    def _draw_detection_results(self, frame, positions):
        """Dibuja resultados de detección"""
        try:
            result_frame = frame.copy()
            for i, (x, y, w, h) in enumerate(positions):
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(result_frame, f"Carta {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self._show_debug_frame("Resultado Final", result_frame)
        except Exception as e:
            print(f"Error dibujando resultados: {e}")
    
    def calibrate_green_range(self, frame):
        """Calibración interactiva de rangos de color"""
        print("🎨 INICIANDO CALIBRACIÓN")
        print("Ajusta los trackbars hasta que solo el tapete verde sea blanco")
        print("Presiona 's' para guardar, 'q' para salir")
        
        # Crear ventana
        cv2.namedWindow('Calibración Verde')
        
        # Valores iniciales
        h_low, h_high = 35, 85
        s_low, s_high = 40, 255
        v_low, v_high = 40, 255
        
        # Crear trackbars
        cv2.createTrackbar('H Low', 'Calibración Verde', h_low, 180, lambda x: None)
        cv2.createTrackbar('H High', 'Calibración Verde', h_high, 180, lambda x: None)
        cv2.createTrackbar('S Low', 'Calibración Verde', s_low, 255, lambda x: None)
        cv2.createTrackbar('S High', 'Calibración Verde', s_high, 255, lambda x: None)
        cv2.createTrackbar('V Low', 'Calibración Verde', v_low, 255, lambda x: None)
        cv2.createTrackbar('V High', 'Calibración Verde', v_high, 255, lambda x: None)
        
        while True:
            try:
                # Obtener valores actuales
                h_low = cv2.getTrackbarPos('H Low', 'Calibración Verde')
                h_high = cv2.getTrackbarPos('H High', 'Calibración Verde')
                s_low = cv2.getTrackbarPos('S Low', 'Calibración Verde')
                s_high = cv2.getTrackbarPos('S High', 'Calibración Verde')
                v_low = cv2.getTrackbarPos('V Low', 'Calibración Verde')
                v_high = cv2.getTrackbarPos('V High', 'Calibración Verde')
                
                # Actualizar rangos
                self.green_lower = np.array([h_low, s_low, v_low])
                self.green_upper = np.array([h_high, s_high, v_high])
                
                # Aplicar máscara
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
                
                # Mostrar resultado
                cv2.imshow('Máscara Calibración', mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("✅ Rangos guardados:")
                    print(f"green_lower = np.array([{h_low}, {s_low}, {v_low}])")
                    print(f"green_upper = np.array([{h_high}, {s_high}, {v_high}])")
                    break
                    
            except Exception as e:
                print(f"Error en calibración: {e}")
                break
        
        cv2.destroyAllWindows()
    
    def preprocess_card(self, card_image):
        """Preprocesa una carta individual para reconocimiento"""
        try:
            if card_image is None or card_image.size == 0:
                return None
                
            # Redimensionar a tamaño estándar
            card_std = cv2.resize(card_image, (self.card_width, self.card_height))
            
            # Mejorar contraste
            lab = cv2.cvtColor(card_std, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0).apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Reducir ruido
            denoised = cv2.medianBlur(enhanced, 3)
            
            return denoised
            
        except Exception as e:
            print(f"Error preprocesando carta: {e}")
            return card_image

# =============================================================================
# FUNCIONES DE PRUEBA Y DEBUG
# =============================================================================

def test_with_image(image_path, calibrate=False):
    """Prueba el detector con una imagen estática"""
    print(f"🚀 Probando detección con: {image_path}")
    
    # Cargar imagen
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ No se pudo cargar la imagen: {image_path}")
        return
    
    # Crear detector en modo debug
    detector = IVCardDetector(debug_mode=True)
    
    # Calibrar si se solicita
    if calibrate:
        detector.calibrate_green_range(frame)
    
    # Detectar cartas
    cards, positions = detector.detect_cards(frame)
    
    print(f"✅ Cartas detectadas: {len(cards)}")
    
    # Mostrar cada carta detectada
    for i, card in enumerate(cards):
        if card is not None and card.size > 0:
            cv2.imshow(f'Carta {i+1}', card)
            # Guardar imagen de carta
            cv2.imwrite(f'carta_detectada_{i+1}.jpg', card)
    
    # Mostrar resultado final
    result_frame = frame.copy()
    for i, (x, y, w, h) in enumerate(positions):
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result_frame, f"Carta {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("RESULTADO FINAL", result_frame)
    cv2.imwrite('resultado_deteccion.jpg', result_frame)
    
    print("Presiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return len(cards)

def test_with_camera():
    """Prueba el detector con la cámara en vivo"""
    print("🎥 Iniciando prueba con cámara...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    detector = IVCardDetector(debug_mode=True)
    
    print("Presiona 'c' para calibrar, 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo frame de cámara")
            break
        
        # Detectar cartas
        cards, positions = detector.detect_cards(frame)
        
        # Mostrar resultado
        display_frame = frame.copy()
        for i, (x, y, w, h) in enumerate(positions):
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Carta {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Cartas: {len(cards)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Presiona 'c' para calibrar, 'q' para salir", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Detección en Vivo", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            detector.calibrate_green_range(frame)
    
    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
# MAIN PARA PRUEBAS
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("🃏 DETECTOR DE CARTAS POKER")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Probar con imagen específica
        image_path = sys.argv[1]
        calibrate = len(sys.argv) > 2 and sys.argv[2].lower() == 'calibrate'
        test_with_image(image_path, calibrate)
    else:
        # Menú interactivo
        print("1. Probar con imagen (test_camera_0.jpg)")
        print("2. Probar con cámara en vivo")
        print("3. Calibrar con imagen")
        
        choice = input("Selecciona opción (1-3): ").strip()
        
        if choice == '1':
            test_with_image("test_camera_0.jpg")
        elif choice == '2':
            test_with_camera()
        elif choice == '3':
            test_with_image("test_camera_0.jpg", calibrate=True)
        else:
            print("Usar: python card_detector.py [imagen_path] [calibrate]")