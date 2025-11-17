import cv2
import numpy as np
import time

class PokerCardDetector:
    def __init__(self):
        self.green_lower = np.array([35, 50, 50])   # Valores por defecto para verde
        self.green_upper = np.array([85, 255, 255])
        self.min_area = 2000
        self.max_area = 50000
        self.is_calibrated = False
        
    def calibrate_green_color(self, frame):
        """Calibración INTERACTIVA del color verde del tapete"""
        print("🎨 CALIBRACIÓN DEL TAPETE VERDE")
        print("==========================================")
        print("INSTRUCCIONES:")
        print("1. Coloca el tapete verde frente a la cámara")
        print("2. Ajusta los trackbars hasta que SOLO el tapete sea BLANCO")
        print("3. Las cartas deben verse NEGRAS")
        print("4. Presiona 's' para GUARDAR")
        print("5. Presiona 'q' para CANCELAR")
        print("==========================================")
        
        # Crear ventanas
        cv2.namedWindow('Calibracion - Ajusta los trackbars')
        cv2.namedWindow('Resultado - Tapete debe ser BLANCO')
        
        # Valores iniciales amplios
        h_min, h_max = 35, 85
        s_min, s_max = 50, 255
        v_min, v_max = 50, 255
        
        # Crear trackbars
        cv2.createTrackbar('H Min', 'Calibracion - Ajusta los trackbars', h_min, 180, lambda x: None)
        cv2.createTrackbar('H Max', 'Calibracion - Ajusta los trackbars', h_max, 180, lambda x: None)
        cv2.createTrackbar('S Min', 'Calibracion - Ajusta los trackbars', s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', 'Calibracion - Ajusta los trackbars', s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', 'Calibracion - Ajusta los trackbars', v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', 'Calibracion - Ajusta los trackbars', v_max, 255, lambda x: None)
        
        print("🔧 Ajustando parámetros...")
        
        while True:
            # Obtener valores de trackbars
            h_min = cv2.getTrackbarPos('H Min', 'Calibracion - Ajusta los trackbars')
            h_max = cv2.getTrackbarPos('H Max', 'Calibracion - Ajusta los trackbars')
            s_min = cv2.getTrackbarPos('S Min', 'Calibracion - Ajusta los trackbars')
            s_max = cv2.getTrackbarPos('S Max', 'Calibracion - Ajusta los trackbars')
            v_min = cv2.getTrackbarPos('V Min', 'Calibracion - Ajusta los trackbars')
            v_max = cv2.getTrackbarPos('V Max', 'Calibracion - Ajusta los trackbars')
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, 
                             np.array([h_min, s_min, v_min]), 
                             np.array([h_max, s_max, v_max]))
            
            # Mostrar resultado
            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow('Resultado - Tapete debe ser BLANCO', result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("❌ Calibración cancelada")
                cv2.destroyAllWindows()
                return False
            elif key == ord('s'):
                self.green_lower = np.array([h_min, s_min, v_min])
                self.green_upper = np.array([h_max, s_max, v_max])
                self.is_calibrated = True
                print("✅ CALIBRACIÓN EXITOSA!")
                print(f"Valores guardados: H[{h_min}-{h_max}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]")
                cv2.destroyAllWindows()
                return True
        
        cv2.destroyAllWindows()
        return False
    
    def detect_cards(self, frame):
        """Detecta cartas en el frame"""
        if not self.is_calibrated:
            print("⚠️  Primero debes calibrar los colores!")
            return [], []
            
        try:
            # Crear máscara verde
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Mejorar la máscara
            kernel = np.ones((5,5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            
            # Invertir máscara (las cartas serán blancas)
            card_mask = cv2.bitwise_not(green_mask)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cards = []
            positions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_area < area < self.max_area:
                    # Aproximar a polígono
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Buscar rectángulos (4 lados)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Verificar relación de aspecto de carta
                        aspect_ratio = w / h
                        if 0.5 <= aspect_ratio <= 0.8:
                            # Extraer carta con margen
                            margin = 10
                            x1 = max(0, x - margin)
                            y1 = max(0, y - margin)
                            x2 = min(frame.shape[1], x + w + margin)
                            y2 = min(frame.shape[0], y + h + margin)
                            
                            card_img = frame[y1:y2, x1:x2]
                            
                            if card_img.size > 0:
                                cards.append(card_img)
                                positions.append((x1, y1, x2-x1, y2-y1))
            
            return cards, positions
            
        except Exception as e:
            print(f"❌ Error detectando cartas: {e}")
            return [], []
    
    def draw_detection(self, frame, positions, card_info=None):
        """Dibuja los resultados en el frame"""
        result = frame.copy()
        
        for i, (x, y, w, h) in enumerate(positions):
            # Dibujar rectángulo alrededor de la carta
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Dibujar información de la carta
            if card_info and i < len(card_info):
                info_text = card_info[i]
                cv2.putText(result, info_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(result, f"Carta {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result

def main():
    print("🃏 DETECTOR DE CARTAS PÓKER")
    print("=" * 50)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    detector = PokerCardDetector()
    
    print("🚀 INICIANDO SISTEMA...")
    print("Presiona las siguientes teclas:")
    print("  'c' - CALIBRAR color del tapete")
    print("  'd' - Mostrar/ocultar DETALLES")
    print("  'q' - SALIR")
    print("=" * 50)
    
    show_details = True
    calibrated = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo cámara")
            break
        
        display_frame = frame.copy()
        
        if detector.is_calibrated:
            # Detectar cartas
            cards, positions = detector.detect_cards(frame)
            
            # Dibujar detecciones
            display_frame = detector.draw_detection(frame, positions)
            
            # Mostrar información
            cv2.putText(display_frame, f"Cartas detectadas: {len(cards)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if show_details:
                cv2.putText(display_frame, "Sistema CALIBRADO - Listo para detectar", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Presiona 'c' para CALIBRAR el tapete verde", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, "Coloca el tapete verde frente a la camara", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar instrucciones
        cv2.putText(display_frame, "Teclas: 'c'=Calibrar, 'd'=Detalles, 'q'=Salir", 
                   (10, display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Detector de Cartas Poker", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("🔧 Iniciando calibración...")
            success = detector.calibrate_green_color(frame)
            if success:
                calibrated = True
                print("✅ Sistema listo para detectar cartas!")
            else:
                print("❌ Calibración fallida")
        elif key == ord('d'):
            show_details = not show_details
            print(f"🔍 Detalles: {'ACTIVADOS' if show_details else 'DESACTIVADOS'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Sistema cerrado")

if __name__ == "__main__":
    main()