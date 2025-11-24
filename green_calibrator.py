import cv2
import numpy as np
import json
import os

class GreenCarpetCalibrator:
    """Calibrador SIMPLE y EFECTIVO para tapete verde"""
    
    def __init__(self):
        self.green_lower = None
        self.green_upper = None
        self.config_file = "green_calibration.json"
    
    def calibrate_interactive(self, camera_index=0):
        """
        Calibración interactiva PASO A PASO
        Retorna True si se calibró exitosamente
        """
        print("=" * 60)
        print("🎨 CALIBRADOR DE TAPETE VERDE - MODO INTERACTIVO")
        print("=" * 60)
        print("\n📋 INSTRUCCIONES:")
        print("1. Coloca SOLO el tapete verde frente a la cámara")
        print("2. Asegúrate de que esté bien iluminado")
        print("3. NO debe haber cartas ni objetos sobre el tapete")
        print("4. Ajusta los trackbars hasta que:")
        print("   ✅ El tapete verde aparezca COMPLETAMENTE BLANCO")
        print("   ✅ Todo lo demás aparezca NEGRO")
        print("5. Presiona 's' para GUARDAR la calibración")
        print("6. Presiona 'q' para CANCELAR")
        print("=" * 60)
        
        input("\n⏸️  Presiona ENTER cuando el tapete esté listo...")
        
        # Abrir cámara
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {camera_index}")
            return False
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print("\n🎥 Cámara iniciada. Capturando frame de referencia...")
        
        # Capturar frame de referencia
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo capturar imagen")
            cap.release()
            return False
        
        cap.release()
        
        # Guardar frame de referencia
        cv2.imwrite("calibration_reference.jpg", frame)
        print("💾 Frame de referencia guardado: calibration_reference.jpg")
        
        # Iniciar calibración con trackbars
        return self._calibrate_with_trackbars(frame)
    
    def _calibrate_with_trackbars(self, frame):
        """Calibración con trackbars - interfaz mejorada"""
        
        # Crear ventanas
        window_name = "Calibracion - Ajusta hasta que tapete sea BLANCO"
        mask_window = "Resultado - BLANCO=tapete, NEGRO=resto"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
        
        # Valores iniciales MÁS AMPLIOS para capturar diferentes tonos de verde
        h_min, h_max = 25, 95   # Más amplio en Hue
        s_min, s_max = 30, 255  # Saturación desde más bajo
        v_min, v_max = 30, 255  # Valor desde más bajo
        
        # Crear trackbars
        cv2.createTrackbar('H Min', window_name, h_min, 180, lambda x: None)
        cv2.createTrackbar('H Max', window_name, h_max, 180, lambda x: None)
        cv2.createTrackbar('S Min', window_name, s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', window_name, s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', window_name, v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', window_name, v_max, 255, lambda x: None)
        
        print("\n🔧 Ajustando parámetros. Ventanas abiertas...")
        print("💡 TIPS:")
        print("   - Empieza ajustando H (Hue) para el tono de verde")
        print("   - Luego ajusta S (Saturación) para la intensidad")
        print("   - Finalmente ajusta V (Valor) para el brillo")
        
        calibrated = False
        
        while True:
            # Leer valores de trackbars
            h_min = cv2.getTrackbarPos('H Min', window_name)
            h_max = cv2.getTrackbarPos('H Max', window_name)
            s_min = cv2.getTrackbarPos('S Min', window_name)
            s_max = cv2.getTrackbarPos('S Max', window_name)
            v_min = cv2.getTrackbarPos('V Min', window_name)
            v_max = cv2.getTrackbarPos('V Max', window_name)
            
            # Crear rangos
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Aplicar operaciones morfológicas para limpiar
            kernel = np.ones((5, 5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
            
            # Calcular porcentaje de píxeles blancos
            white_pixels = cv2.countNonZero(mask_clean)
            total_pixels = mask_clean.shape[0] * mask_clean.shape[1]
            white_percentage = (white_pixels / total_pixels) * 100
            
            # Crear imagen con información
            info_frame = frame.copy()
            cv2.putText(info_frame, f"Tapete detectado: {white_percentage:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(info_frame, "Ajusta trackbars. 's'=Guardar, 'q'=Cancelar", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, f"H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}]", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Mostrar ventanas
            cv2.imshow(window_name, info_frame)
            cv2.imshow(mask_window, mask_clean)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n❌ Calibración cancelada por el usuario")
                calibrated = False
                break
                
            elif key == ord('s'):
                # Validar que hay suficiente tapete detectado
                if white_percentage < 20:
                    print(f"\n⚠️  Solo se detecta {white_percentage:.1f}% de tapete verde")
                    print("   Ajusta los valores para detectar más área verde")
                    continue
                
                if white_percentage > 95:
                    print(f"\n⚠️  Se está detectando {white_percentage:.1f}% como verde")
                    print("   Parece que estás detectando demasiado. Ajusta los valores.")
                    continue
                
                # Guardar calibración
                self.green_lower = lower
                self.green_upper = upper
                
                print("\n✅ CALIBRACIÓN EXITOSA!")
                print(f"📊 Tapete detectado: {white_percentage:.1f}%")
                print(f"📝 Valores guardados:")
                print(f"   H (Hue):        [{h_min} - {h_max}]")
                print(f"   S (Saturación): [{s_min} - {s_max}]")
                print(f"   V (Valor):      [{v_min} - {v_max}]")
                
                # Guardar en archivo
                self.save_calibration()
                calibrated = True
                break
        
        cv2.destroyAllWindows()
        return calibrated
    
    def save_calibration(self):
        """Guarda la calibración en archivo JSON"""
        if self.green_lower is None or self.green_upper is None:
            print("⚠️  No hay calibración para guardar")
            return False
        
        config = {
            'green_lower': self.green_lower.tolist(),
            'green_upper': self.green_upper.tolist(),
            'timestamp': str(np.datetime64('now'))
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"💾 Calibración guardada en: {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ Error guardando calibración: {e}")
            return False
    
    def load_calibration(self):
        """Carga calibración desde archivo"""
        if not os.path.exists(self.config_file):
            print(f"⚠️  No existe archivo de calibración: {self.config_file}")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.green_lower = np.array(config['green_lower'])
            self.green_upper = np.array(config['green_upper'])
            
            print("✅ Calibración cargada exitosamente")
            print(f"   Lower: {self.green_lower}")
            print(f"   Upper: {self.green_upper}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando calibración: {e}")
            return False
    
    def test_calibration(self, camera_index=0):
        """Prueba la calibración en tiempo real"""
        if self.green_lower is None or self.green_upper is None:
            print("❌ Primero debes calibrar o cargar una calibración")
            return
        
        print("\n🧪 PROBANDO CALIBRACIÓN")
        print("Presiona 'q' para salir")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Limpiar máscara
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Invertir para ver cartas
            cards_mask = cv2.bitwise_not(mask)
            
            # Encontrar contornos de posibles cartas
            contours, _ = cv2.findContours(cards_mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Dibujar contornos de cartas potenciales
            result = frame.copy()
            card_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 2000 < area < 50000:  # Rango de área de cartas
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.5 <= aspect_ratio <= 0.9:
                        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        card_count += 1
            
            # Mostrar información
            cv2.putText(result, f"Cartas detectadas: {card_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result, "Presiona 'q' para salir", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar ventanas
            cv2.imshow("Test - Deteccion de Cartas", result)
            cv2.imshow("Mascara - Verde=Tapete", mask)
            cv2.imshow("Cartas - Blanco=Posibles cartas", cards_mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test finalizado")


def main():
    """Menú principal del calibrador"""
    calibrator = GreenCarpetCalibrator()
    
    while True:
        print("\n" + "=" * 60)
        print("🎯 CALIBRADOR DE TAPETE VERDE - MENÚ PRINCIPAL")
        print("=" * 60)
        print("1. ✨ Calibrar tapete (NUEVO)")
        print("2. 📂 Cargar calibración existente")
        print("3. 🧪 Probar calibración")
        print("4. ❌ Salir")
        print("=" * 60)
        
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            # Calibrar
            camera = input("Índice de cámara (default=0): ").strip()
            camera_idx = int(camera) if camera else 0
            
            success = calibrator.calibrate_interactive(camera_idx)
            
            if success:
                print("\n🎉 ¡Calibración completada!")
                test = input("\n¿Quieres probar la calibración ahora? (s/n): ").lower()
                if test == 's':
                    calibrator.test_calibration(camera_idx)
        
        elif choice == '2':
            # Cargar calibración
            success = calibrator.load_calibration()
            
            if success:
                test = input("\n¿Quieres probar la calibración? (s/n): ").lower()
                if test == 's':
                    camera = input("Índice de cámara (default=0): ").strip()
                    camera_idx = int(camera) if camera else 0
                    calibrator.test_calibration(camera_idx)
        
        elif choice == '3':
            # Probar calibración
            if calibrator.green_lower is None:
                print("\n⚠️  Primero debes calibrar o cargar una calibración")
            else:
                camera = input("Índice de cámara (default=0): ").strip()
                camera_idx = int(camera) if camera else 0
                calibrator.test_calibration(camera_idx)
        
        elif choice == '4':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción no válida")


if __name__ == "__main__":
    main()