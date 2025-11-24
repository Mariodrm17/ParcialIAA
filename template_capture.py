import cv2
import numpy as np
import os
import json
from datetime import datetime

class CardTemplateCapture:
    """Sistema para capturar y guardar plantillas de cartas"""
    
    def __init__(self):
        self.templates_dir = "card_templates"
        self.config_file = "templates_config.json"
        
        # Crear directorio si no existe
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
            print(f"✅ Directorio creado: {self.templates_dir}")
        
        # Cartas de una baraja estándar
        self.numbers = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.suits = {
            'H': 'Corazones',
            'D': 'Diamantes', 
            'C': 'Treboles',
            'S': 'Picas'
        }
        
        # Cargar configuración existente
        self.load_config()
    
    def load_config(self):
        """Carga configuración de plantillas existentes"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                print(f"✅ Configuración cargada: {len(self.config.get('templates', []))} plantillas")
            except:
                self.config = {'templates': []}
        else:
            self.config = {'templates': []}
    
    def save_config(self):
        """Guarda configuración"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print("✅ Configuración guardada")
        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")
    
    def capture_template_interactive(self, camera_index=0):
        """Modo interactivo para capturar plantillas"""
        
        print("\n" + "=" * 70)
        print("📸 SISTEMA DE CAPTURA DE PLANTILLAS")
        print("=" * 70)
        print("\n📋 INSTRUCCIONES:")
        print("1. Coloca UNA SOLA CARTA sobre el tapete verde")
        print("2. Asegúrate de que la carta esté bien visible y enfocada")
        print("3. La carta debe estar completamente dentro del cuadro verde")
        print("4. Presiona ESPACIO para capturar la plantilla")
        print("5. Presiona 'q' para salir")
        print("=" * 70)
        
        input("\n⏸️  Presiona ENTER para iniciar la cámara...")
        
        # Cargar calibración del tapete
        if not os.path.exists("green_calibration.json"):
            print("❌ No se encontró calibración del tapete verde")
            print("   Ejecuta primero: python green_calibrator.py")
            return
        
        with open("green_calibration.json", 'r') as f:
            calibration = json.load(f)
        
        green_lower = np.array(calibration['green_lower'])
        green_upper = np.array(calibration['green_upper'])
        
        # Abrir cámara
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print("\n🎥 Cámara iniciada. Coloca una carta...")
        
        captured_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Detectar carta
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Limpiar máscara
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            
            # Invertir (cartas = blanco)
            cards_mask = cv2.bitwise_not(green_mask)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(cards_mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            card_detected = False
            card_img = None
            card_position = None
            
            # Buscar la carta más grande
            largest_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 2000 < area < 50000 and area > largest_area:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        if 0.5 <= aspect_ratio <= 0.9:
                            largest_area = area
                            card_detected = True
                            
                            # Dibujar rectángulo verde si es válida
                            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            
                            # Extraer carta
                            margin = 15
                            x1 = max(0, x - margin)
                            y1 = max(0, y - margin)
                            x2 = min(frame.shape[1], x + w + margin)
                            y2 = min(frame.shape[0], y + h + margin)
                            
                            card_img = frame[y1:y2, x1:x2]
                            card_position = (x, y, w, h)
                            
                            # Mostrar mensaje
                            cv2.putText(display, "CARTA DETECTADA - Presiona ESPACIO", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 255, 0), 2)
            
            # Información en pantalla
            if card_detected:
                cv2.putText(display, "ESPACIO = Capturar | q = Salir", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display, f"Plantillas capturadas: {captured_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(display, "Coloca UNA carta sobre el tapete verde", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display, f"Plantillas capturadas: {captured_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Captura de Plantillas", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' ') and card_detected and card_img is not None:
                # Capturar plantilla
                self.save_template(card_img, frame)
                captured_count += 1
                print(f"✅ Plantilla #{captured_count} capturada")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎉 Captura finalizada. Total plantillas: {captured_count}")
    
    def save_template(self, card_img, full_frame):
        """Guarda una plantilla con información del usuario"""
        
        # Crear ventana para mostrar la carta capturada
        card_display = cv2.resize(card_img, (400, 600))
        
        cv2.imshow("Carta Capturada", card_display)
        cv2.waitKey(1)
        
        print("\n" + "=" * 50)
        print("📝 INFORMACIÓN DE LA CARTA")
        print("=" * 50)
        
        # Mostrar opciones de números
        print("\nNÚMEROS DISPONIBLES:")
        for i, num in enumerate(self.numbers, 1):
            print(f"{i:2d}. {num}", end="  ")
            if i % 6 == 0:
                print()
        print()
        
        # Solicitar número
        while True:
            try:
                num_choice = input("\nSelecciona el NÚMERO (1-13): ").strip()
                num_idx = int(num_choice) - 1
                if 0 <= num_idx < len(self.numbers):
                    number = self.numbers[num_idx]
                    break
                else:
                    print("❌ Opción inválida")
            except:
                print("❌ Entrada inválida")
        
        # Mostrar opciones de palos
        print("\nPALOS DISPONIBLES:")
        print("1. ♥ Corazones (H)")
        print("2. ♦ Diamantes (D)")
        print("3. ♣ Tréboles (C)")
        print("4. ♠ Picas (S)")
        
        suit_map = {1: 'H', 2: 'D', 3: 'C', 4: 'S'}
        
        # Solicitar palo
        while True:
            try:
                suit_choice = input("\nSelecciona el PALO (1-4): ").strip()
                suit_idx = int(suit_choice)
                if suit_idx in suit_map:
                    suit = suit_map[suit_idx]
                    break
                else:
                    print("❌ Opción inválida")
            except:
                print("❌ Entrada inválida")
        
        # Crear nombre de archivo
        card_name = f"{number}_{suit}"
        suit_name = self.suits[suit]
        full_name = f"{number} de {suit_name}"
        
        # Generar nombre único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{card_name}_{timestamp}.jpg"
        filepath = os.path.join(self.templates_dir, filename)
        
        # Guardar imagen
        cv2.imwrite(filepath, card_img)
        
        # Guardar frame completo para referencia
        ref_filename = f"{card_name}_{timestamp}_full.jpg"
        ref_filepath = os.path.join(self.templates_dir, ref_filename)
        cv2.imwrite(ref_filepath, full_frame)
        
        # Actualizar configuración
        template_info = {
            'filename': filename,
            'card_name': card_name,
            'full_name': full_name,
            'number': number,
            'suit': suit,
            'timestamp': timestamp
        }
        
        self.config['templates'].append(template_info)
        self.save_config()
        
        print(f"\n✅ Plantilla guardada: {full_name}")
        print(f"📁 Archivo: {filename}")
        
        cv2.destroyWindow("Carta Capturada")
    
    def list_templates(self):
        """Lista todas las plantillas guardadas"""
        print("\n" + "=" * 50)
        print("📋 PLANTILLAS GUARDADAS")
        print("=" * 50)
        
        if not self.config['templates']:
            print("❌ No hay plantillas guardadas")
            return
        
        # Agrupar por palo
        by_suit = {'H': [], 'D': [], 'C': [], 'S': []}
        for template in self.config['templates']:
            by_suit[template['suit']].append(template)
        
        for suit_code, suit_name in self.suits.items():
            templates = by_suit[suit_code]
            if templates:
                print(f"\n{suit_name}:")
                for t in sorted(templates, key=lambda x: self.numbers.index(x['number'])):
                    print(f"  ✅ {t['full_name']}")
        
        print(f"\n📊 Total: {len(self.config['templates'])} plantillas")
    
    def delete_template(self, card_name):
        """Elimina una plantilla"""
        templates_to_keep = []
        deleted = False
        
        for template in self.config['templates']:
            if template['card_name'] != card_name:
                templates_to_keep.append(template)
            else:
                # Eliminar archivo
                filepath = os.path.join(self.templates_dir, template['filename'])
                if os.path.exists(filepath):
                    os.remove(filepath)
                deleted = True
        
        if deleted:
            self.config['templates'] = templates_to_keep
            self.save_config()
            print(f"✅ Plantilla {card_name} eliminada")
        else:
            print(f"❌ Plantilla {card_name} no encontrada")


def main():
    """Menú principal"""
    capture = CardTemplateCapture()
    
    while True:
        print("\n" + "=" * 60)
        print("🃏 SISTEMA DE CAPTURA DE PLANTILLAS DE CARTAS")
        print("=" * 60)
        print("1. 📸 Capturar plantillas (modo interactivo)")
        print("2. 📋 Listar plantillas guardadas")
        print("3. 🗑️  Eliminar plantilla")
        print("4. ❌ Salir")
        print("=" * 60)
        
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            camera = input("Índice de cámara (default=0): ").strip()
            camera_idx = int(camera) if camera else 0
            
            print("\n💡 CONSEJO: Ten preparadas las cartas que quieras capturar")
            print("           Puedes capturar varias en la misma sesión")
            
            input("\n⏸️  Presiona ENTER para continuar...")
            capture.capture_template_interactive(camera_idx)
        
        elif choice == '2':
            capture.list_templates()
        
        elif choice == '3':
            capture.list_templates()
            card_name = input("\nNombre de la carta a eliminar (ej: A_H): ").strip()
            if card_name:
                capture.delete_template(card_name)
        
        elif choice == '4':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción no válida")


if __name__ == "__main__":
    main()