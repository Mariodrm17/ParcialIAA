import cv2
import numpy as np

class CardDetector:
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        
        # Rangos HSV MÁS AMPLIOS - se ajustarán en calibración
        self.green_lower = np.array([30, 30, 30])    # Más bajo
        self.green_upper = np.array([90, 255, 255])  # Más alto
        
        # Parámetros de detección más permisivos
        self.min_area = 1500
        self.max_area = 50000
        
    def detect_cards(self, frame):
        """Detección SIMPLIFICADA y ROBUSTA"""
        try:
            if frame is None:
                return [], []
                
            original = frame.copy()
            cards = []
            positions = []
            
            # 1. CONVERTIR A HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 2. CREAR MÁSCARA VERDE
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # 3. MEJORAR LA MÁSCARA
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            if self.debug_mode:
                cv2.imshow("1 - Máscara Verde", mask)
                print(f"Píxeles verdes detectados: {cv2.countNonZero(mask)}")
            
            # 4. ENCONTRAR CONTORNOS
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if self.debug_mode:
                print(f"Contornos encontrados: {len(contours)}")
            
            # 5. FILTRAR CONTORNOS QUE PODRÍAN SER CARTAS
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por área
                if area < self.min_area or area > self.max_area:
                    continue
                
                # Simplificar contorno
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Buscar formas con 4 lados (cartas)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Verificar relación de aspecto de carta
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 0.8:  # Relación típica de cartas
                        # Extraer la carta con margen
                        margin = 15
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(frame.shape[1], x + w + margin)
                        y2 = min(frame.shape[0], y + h + margin)
                        
                        card_img = original[y1:y2, x1:x2]
                        
                        if card_img.size > 0:
                            cards.append(card_img)
                            positions.append((x1, y1, x2-x1, y2-y1))
                            
                            if self.debug_mode:
                                print(f"Carta detectada: {x1}, {y1}, {w}, {h}, área: {area}")
            
            if self.debug_mode:
                # Mostrar resultado
                result = original.copy()
                for (x, y, w, h) in positions:
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(result, "CARTA", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("2 - Cartas Detectadas", result)
                cv2.waitKey(1)
            
            return cards, positions
            
        except Exception as e:
            print(f"Error en detección: {e}")
            return [], []
    
    def calibrate_color(self, frame):
        """CALIBRACIÓN INTERACTIVA SIMPLE"""
        print("🎨 CALIBRADOR DE COLOR")
        print("Ajusta los trackbars hasta que SOLO el tapete verde sea BLANCO")
        print("Presiona 's' para GUARDAR, 'q' para SALIR")
        
        # Crear ventanas
        cv2.namedWindow('Calibracion')
        cv2.namedWindow('Mascara')
        
        # Valores iniciales MÁS AMPLIOS
        h_min, h_max = 30, 90
        s_min, s_max = 30, 255
        v_min, v_max = 30, 255
        
        # Crear trackbars
        cv2.createTrackbar('H Min', 'Calibracion', h_min, 180, lambda x: None)
        cv2.createTrackbar('H Max', 'Calibracion', h_max, 180, lambda x: None)
        cv2.createTrackbar('S Min', 'Calibracion', s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', 'Calibracion', s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', 'Calibracion', v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', 'Calibracion', v_max, 255, lambda x: None)
        
        while True:
            # Obtener valores actuales
            h_min = cv2.getTrackbarPos('H Min', 'Calibracion')
            h_max = cv2.getTrackbarPos('H Max', 'Calibracion')
            s_min = cv2.getTrackbarPos('S Min', 'Calibracion')
            s_max = cv2.getTrackbarPos('S Max', 'Calibracion')
            v_min = cv2.getTrackbarPos('V Min', 'Calibracion')
            v_max = cv2.getTrackbarPos('V Max', 'Calibracion')
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, 
                             np.array([h_min, s_min, v_min]), 
                             np.array([h_max, s_max, v_max]))
            
            # Mostrar máscara
            cv2.imshow('Mascara', mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.green_lower = np.array([h_min, s_min, v_min])
                self.green_upper = np.array([h_max, s_max, v_max])
                print("✅ VALORES GUARDADOS:")
                print(f"LOWER: [{h_min}, {s_min}, {v_min}]")
                print(f"UPPER: [{h_max}, {s_max}, {v_max}]")
                break
        
        cv2.destroyAllWindows()
        return self.green_lower, self.green_upper

# =============================================================================
# PRUEBAS INMEDIATAS
# =============================================================================

def prueba_rapida():
    """Prueba RÁPIDA con imagen"""
    print("🚀 PRUEBA RÁPIDA DE DETECCIÓN")
    
    # Cargar imagen
    frame = cv2.imread("test_camera_0.jpg")
    if frame is None:
        print("❌ No se pudo cargar test_camera_0.jpg")
        print("📸 Por favor, asegúrate de que la imagen existe")
        return
    
    detector = CardDetector(debug_mode=True)
    
    # Preguntar si calibrar
    respuesta = input("¿Quieres calibrar los colores? (s/n): ").lower()
    if respuesta == 's':
        detector.calibrate_color(frame)
    
    # Detectar cartas
    print("🔍 Buscando cartas...")
    cards, positions = detector.detect_cards(frame)
    
    print(f"🎯 Resultado: {len(cards)} cartas detectadas")
    
    # Mostrar resultados finales
    result_frame = frame.copy()
    for i, (x, y, w, h) in enumerate(positions):
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(result_frame, f"Carta {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("RESULTADO FINAL", result_frame)
    
    # Guardar imagen con resultados
    cv2.imwrite("resultado_deteccion.jpg", result_frame)
    print("💾 Imagen guardada como 'resultado_deteccion.jpg'")
    
    # Mostrar cada carta detectada
    for i, card in enumerate(cards):
        cv2.imshow(f"Carta {i+1}", card)
        cv2.imwrite(f"carta_{i+1}.jpg", card)
        print(f"💾 Carta {i+1} guardada como 'carta_{i+1}.jpg'")
    
    print("Presiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prueba_camara():
    """Prueba con cámara en vivo"""
    print("🎥 INICIANDO CÁMARA...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = CardDetector(debug_mode=True)
    
    print("Instrucciones:")
    print("- Presiona 'c' para CALIBRAR colores")
    print("- Presiona 'q' para SALIR")
    
    calibrated = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo cámara")
            break
        
        # Detectar cartas
        cards, positions = detector.detect_cards(frame)
        
        # Mostrar información en pantalla
        display = frame.copy()
        cv2.putText(display, f"Cartas: {len(cards)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Presiona 'c' para calibrar, 'q' para salir", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dibujar cartas detectadas
        for (x, y, w, h) in positions:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, "CARTA", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Detector de Cartas - EN VIVO", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("🎨 Iniciando calibración...")
            detector.calibrate_color(frame)
            calibrated = True
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=" * 50)
    print("🃏 DETECTOR DE CARTAS POKER - VERSIÓN SIMPLIFICADA")
    print("=" * 50)
    
    opcion = input("¿Qué quieres hacer?\n1. Probar con imagen\n2. Probar con cámara\nElige (1/2): ").strip()
    
    if opcion == "1":
        prueba_rapida()
    elif opcion == "2":
        prueba_camara()
    else:
        print("❌ Opción no válida")