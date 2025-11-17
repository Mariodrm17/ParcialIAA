import cv2
import numpy as np

class ColorCalibrator:
    def __init__(self):
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
    
    def calibrate_from_image(self, image_path):
        """Calibra los colores desde una imagen estática"""
        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ No se pudo cargar la imagen")
            return
        
        return self.calibrate_live(frame)
    
    def calibrate_live(self, frame):
        """Calibración interactiva con la cámara en vivo - VERSIÓN CORREGIDA"""
        print("🎨 CALIBRACIÓN DE COLORES")
        print("1. Ajusta los trackbars hasta que solo el tapete verde sea blanco")
        print("2. Presiona 's' para guardar, 'q' para salir")
        
        # Crear ventanas
        cv2.namedWindow('Calibración Verde')
        cv2.namedWindow('Resultado')
        
        # Inicializar valores de trackbar
        h_low, h_high = 35, 85
        s_low, s_high = 40, 255
        v_low, v_high = 40, 255
        
        # Crear trackbars - MÉTODO SIMPLIFICADO Y CORREGIDO
        cv2.createTrackbar('H Low', 'Calibración Verde', h_low, 180, lambda x: None)
        cv2.createTrackbar('H High', 'Calibración Verde', h_high, 180, lambda x: None)
        cv2.createTrackbar('S Low', 'Calibración Verde', s_low, 255, lambda x: None)
        cv2.createTrackbar('S High', 'Calibración Verde', s_high, 255, lambda x: None)
        cv2.createTrackbar('V Low', 'Calibración Verde', v_low, 255, lambda x: None)
        cv2.createTrackbar('V High', 'Calibración Verde', v_high, 255, lambda x: None)
        
        while True:
            # Obtener posiciones actuales de los trackbars
            h_low = cv2.getTrackbarPos('H Low', 'Calibración Verde')
            h_high = cv2.getTrackbarPos('H High', 'Calibración Verde')
            s_low = cv2.getTrackbarPos('S Low', 'Calibración Verde')
            s_high = cv2.getTrackbarPos('S High', 'Calibración Verde')
            v_low = cv2.getTrackbarPos('V Low', 'Calibración Verde')
            v_high = cv2.getTrackbarPos('V High', 'Calibración Verde')
            
            # Asegurar que low <= high
            if h_low > h_high:
                h_low = h_high
                cv2.setTrackbarPos('H Low', 'Calibración Verde', h_low)
            if s_low > s_high:
                s_low = s_high
                cv2.setTrackbarPos('S Low', 'Calibración Verde', s_low)
            if v_low > v_high:
                v_low = v_high
                cv2.setTrackbarPos('V Low', 'Calibración Verde', v_low)
            
            # Actualizar rangos
            self.green_lower = np.array([h_low, s_low, v_low])
            self.green_upper = np.array([h_high, s_high, v_high])
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Mostrar resultados
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Crear imagen combinada para visualización
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([frame, mask_bgr, result])
            
            # Redimensionar si es muy grande
            h, w = combined.shape[:2]
            if w > 1200:
                scale = 1200 / w
                new_h = int(h * scale)
                combined = cv2.resize(combined, (1200, new_h))
            
            cv2.imshow('Resultado', combined)
            
            # Instrucciones en la imagen
            instructions = "Ajusta trackbars. 's'=Guardar, 'q'=Salir"
            cv2.putText(combined, instructions, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Resultado', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("❌ Calibración cancelada")
                break
            elif key == ord('s'):
                print("✅ Valores guardados:")
                print(f"green_lower = np.array([{h_low}, {s_low}, {v_low}])")
                print(f"green_upper = np.array([{h_high}, {s_high}, {v_high}])")
                break
        
        cv2.destroyAllWindows()
        return self.green_lower, self.green_upper

# Versión simplificada para prueba rápida
def quick_calibrate(image_path):
    """Función simple para calibrar rápidamente"""
    calibrator = ColorCalibrator()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ No se pudo cargar {image_path}")
        return None, None
    
    print("🔧 CALIBRADOR RÁPIDO")
    print("Ajusta los trackbars para que solo el tapete verde se vea blanco")
    print("Luego presiona 's' para guardar o 'q' para salir")
    
    return calibrator.calibrate_live(frame)

if __name__ == "__main__":
    # Probar con la imagen de prueba
    image_path = "test_camera_0.jpg"
    
    lower, upper = quick_calibrate(image_path)
    
    if lower is not None and upper is not None:
        print("\n🎯 VALORES FINALES:")
        print(f"LOWER: [{lower[0]}, {lower[1]}, {lower[2]}]")
        print(f"UPPER: [{upper[0]}, {upper[1]}, {upper[2]}]")
        
        # Probar los valores
        frame = cv2.imread(image_path)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        cv2.imshow('Máscara Final', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()