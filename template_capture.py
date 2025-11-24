import cv2
import numpy as np
import os
import json
from datetime import datetime

class CardTemplateCapture:
    """Sistema mejorado para capturar ESQUINAS y PALOS por separado"""
    
    def __init__(self):
        self.templates_dir = "card_templates"
        self.corners_dir = os.path.join(self.templates_dir, "corners")
        self.suits_dir = os.path.join(self.templates_dir, "suits")
        self.config_file = "templates_config.json"
        
        # Crear directorios
        for directory in [self.templates_dir, self.corners_dir, self.suits_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"✅ Directorio creado: {directory}")
        
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
                self.config = {'templates': [], 'corners': [], 'suits': []}
        else:
            self.config = {'templates': [], 'corners': [], 'suits': []}
    
    def save_config(self):
        """Guarda configuración"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print("✅ Configuración guardada")
        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")
    
    def extract_corner_region(self, card_img, show_preview=True):
        """
        Extrae la esquina superior izquierda de forma INTELIGENTE
        Busca el área donde está el número y palo
        """
        height, width = card_img.shape[:2]
        
        # MEJORADO: Ajustar según tamaño real de cartas
        # Esquina típica: ~35% alto x 30% ancho
        corner_h = int(height * 0.35)  # Aumentado de 0.30 a 0.35
        corner_w = int(width * 0.30)   # Aumentado de 0.25 a 0.30
        
        # Extraer esquina
        corner = card_img[0:corner_h, 0:corner_w].copy()
        
        if show_preview:
            # Mostrar vista previa con marco
            preview = card_img.copy()
            cv2.rectangle(preview, (0, 0), (corner_w, corner_h), (0, 255, 0), 3)
            cv2.putText(preview, "Esquina a guardar", (10, corner_h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar ambas vistas
            display = np.hstack([
                cv2.resize(preview, (300, 450)),
                cv2.resize(corner, (150, 225))
            ])
            cv2.imshow("Vista Previa - Esquina", display)
            cv2.waitKey(500)
        
        return corner
    
    def extract_suit_region(self, card_img, show_preview=True):
        """
        Extrae SOLO la región del palo (símbolo) de forma más precisa
        """
        height, width = card_img.shape[:2]
        
        # El palo está típicamente en la parte superior, debajo del número
        # Aproximadamente: desde 20% hasta 50% de la altura, centrado
        suit_y1 = int(height * 0.15)  # Empieza más abajo del número
        suit_y2 = int(height * 0.45)  # Hasta casi la mitad
        suit_x1 = int(width * 0.05)   # Un poco de margen izquierdo
        suit_x2 = int(width * 0.35)   # ~30% del ancho
        
        # Extraer región del palo
        suit_region = card_img[suit_y1:suit_y2, suit_x1:suit_x2].copy()
        
        if show_preview:
            # Mostrar vista previa
            preview = card_img.copy()
            cv2.rectangle(preview, (suit_x1, suit_y1), (suit_x2, suit_y2), (255, 0, 0), 3)
            cv2.putText(preview, "Palo a guardar", (suit_x1, suit_y2 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            display = np.hstack([
                cv2.resize(preview, (300, 450)),
                cv2.resize(suit_region, (150, 225))
            ])
            cv2.imshow("Vista Previa - Palo", display)
            cv2.waitKey(500)
        
        return suit_region
    
    def extract_number_region(self, card_img, show_preview=True):
        """
        Extrae SOLO la región del número/letra (más pequeña y precisa)
        """
        height, width = card_img.shape[:2]
        
        # El número está en la parte superior izquierda
        # Aproximadamente: primeros 25% altura x 25% ancho
        num_h = int(height * 0.25)
        num_w = int(width * 0.25)
        
        number_region = card_img[0:num_h, 0:num_w].copy()
        
        if show_preview:
            preview = card_img.copy()
            cv2.rectangle(preview, (0, 0), (num_w, num_h), (0, 255, 255), 3)
            cv2.putText(preview, "Numero", (10, num_h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            display = np.hstack([
                cv2.resize(preview, (300, 450)),
                cv2.resize(number_region, (150, 150))
            ])
            cv2.imshow("Vista Previa - Numero", display)
            cv2.waitKey(500)
        
        return number_region
    
    def capture_template_interactive(self, camera_index=0):
        """Modo interactivo mejorado para capturar plantillas con esquinas y palos"""
        
        print("\n" + "=" * 70)
        print("📸 SISTEMA DE CAPTURA MEJORADO - ESQUINAS Y PALOS")
        print("=" * 70)
        print("\n📋 INSTRUCCIONES:")
        print("1. Coloca UNA carta sobre el tapete verde")
        print("2. Asegúrate de que esté bien iluminada y enfocada")
        print("3. La carta debe estar VERTICAL (esquina arriba-izquierda)")
        print("4. Presiona ESPACIO para capturar")
        print("5. El sistema guardará:")
        print("   • Carta completa")
        print("   • Esquina (número + palo juntos)")
        print("   • Solo número")
        print("   • Solo palo")
        print("=" * 70)
        
        input("\n⏸️  Presiona ENTER para iniciar...")
        
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
        print("💡 TIP: La carta debe estar lo más recta posible")
        
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
                            
                            # Dibujar rectángulo verde
                            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            
                            # Marcar regiones que se capturarán
                            # Esquina completa
                            corner_h = int(h * 0.35)
                            corner_w = int(w * 0.30)
                            cv2.rectangle(display, (x, y), (x+corner_w, y+corner_h), (0, 255, 255), 2)
                            cv2.putText(display, "Esquina", (x+5, y+15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            # Número
                            num_h = int(h * 0.25)
                            num_w = int(w * 0.25)
                            cv2.rectangle(display, (x, y), (x+num_w, y+num_h), (255, 255, 0), 2)
                            cv2.putText(display, "Num", (x+5, y+num_h-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            # Palo
                            suit_y1 = y + int(h * 0.15)
                            suit_y2 = y + int(h * 0.45)
                            suit_x1 = x + int(w * 0.05)
                            suit_x2 = x + int(w * 0.35)
                            cv2.rectangle(display, (suit_x1, suit_y1), (suit_x2, suit_y2), (255, 0, 255), 2)
                            cv2.putText(display, "Palo", (suit_x1+5, suit_y2-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                            
                            # Extraer carta
                            margin = 15
                            x1 = max(0, x - margin)
                            y1 = max(0, y - margin)
                            x2 = min(frame.shape[1], x + w + margin)
                            y2 = min(frame.shape[0], y + h + margin)
                            
                            card_img = frame[y1:y2, x1:x2]
                            card_position = (x, y, w, h)
                            
                            # Mostrar mensaje
                            cv2.putText(display, "CARTA OK - Presiona ESPACIO", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 255, 0), 2)
            
            # Información en pantalla
            if card_detected:
                cv2.putText(display, "ESPACIO = Capturar | q = Salir", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display, f"Capturas: {captured_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display, "Amarillo=Num, Cyan=Esquina, Magenta=Palo", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            else:
                cv2.putText(display, "Coloca UNA carta VERTICAL sobre el tapete", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display, f"Capturas: {captured_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Captura de Plantillas - MEJORADO", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' ') and card_detected and card_img is not None:
                # Capturar con regiones separadas
                self.save_template_with_regions(card_img, frame)
                captured_count += 1
                print(f"✅ Plantilla #{captured_count} capturada")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎉 Captura finalizada. Total: {captured_count} plantillas")
    
    def save_template_with_regions(self, card_img, full_frame):
        """Guarda plantilla con TODAS las regiones extraídas"""
        
        print("\n" + "=" * 50)
        print("📝 INFORMACIÓN DE LA CARTA")
        print("=" * 50)
        
        # Extraer regiones
        corner = self.extract_corner_region(card_img, show_preview=True)
        suit_region = self.extract_suit_region(card_img, show_preview=True)
        number_region = self.extract_number_region(card_img, show_preview=True)
        
        # Mostrar todas las regiones juntas
        try:
            combined = np.hstack([
                cv2.resize(card_img, (200, 300)),
                cv2.resize(corner, (100, 150)),
                cv2.resize(number_region, (75, 75)),
                cv2.resize(suit_region, (75, 112))
            ])
            cv2.imshow("Resumen: Completa | Esquina | Numero | Palo", combined)
            cv2.waitKey(1000)
        except:
            pass
        
        # Solicitar información de la carta
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
        
        # Solicitar palo
        print("\nPALOS DISPONIBLES:")
        print("1. ♥ Corazones (H)")
        print("2. ♦ Diamantes (D)")
        print("3. ♣ Tréboles (C)")
        print("4. ♠ Picas (S)")
        
        suit_map = {1: 'H', 2: 'D', 3: 'C', 4: 'S'}
        
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
        
        # Crear nombres de archivo
        card_name = f"{number}_{suit}"
        suit_name = self.suits[suit]
        full_name = f"{number} de {suit_name}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar TODAS las imágenes
        files_saved = {}
        
        # 1. Carta completa
        full_filename = f"{card_name}_{timestamp}_full.jpg"
        full_filepath = os.path.join(self.templates_dir, full_filename)
        cv2.imwrite(full_filepath, card_img)
        files_saved['full'] = full_filename
        
        # 2. Esquina completa (número + palo)
        corner_filename = f"{card_name}_{timestamp}_corner.jpg"
        corner_filepath = os.path.join(self.corners_dir, corner_filename)
        cv2.imwrite(corner_filepath, corner)
        files_saved['corner'] = corner_filename
        
        # 3. Solo número
        number_filename = f"{card_name}_{timestamp}_number.jpg"
        number_filepath = os.path.join(self.corners_dir, number_filename)
        cv2.imwrite(number_filepath, number_region)
        files_saved['number'] = number_filename
        
        # 4. Solo palo
        suit_filename = f"{card_name}_{timestamp}_suit.jpg"
        suit_filepath = os.path.join(self.suits_dir, suit_filename)
        cv2.imwrite(suit_filepath, suit_region)
        files_saved['suit'] = suit_filename
        
        # 5. Frame completo de referencia
        ref_filename = f"{card_name}_{timestamp}_reference.jpg"
        ref_filepath = os.path.join(self.templates_dir, ref_filename)
        cv2.imwrite(ref_filepath, full_frame)
        files_saved['reference'] = ref_filename
        
        # Actualizar configuración
        template_info = {
            'card_name': card_name,
            'full_name': full_name,
            'number': number,
            'suit': suit,
            'timestamp': timestamp,
            'files': files_saved
        }
        
        self.config['templates'].append(template_info)
        self.save_config()
        
        print(f"\n✅ Plantilla guardada: {full_name}")
        print(f"📁 Archivos creados:")
        print(f"   • Completa:   {full_filename}")
        print(f"   • Esquina:    {corner_filename}")
        print(f"   • Número:     {number_filename}")
        print(f"   • Palo:       {suit_filename}")
        print(f"   • Referencia: {ref_filename}")
        
        cv2.destroyAllWindows()
    
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
        
        total = 0
        for suit_code, suit_name in self.suits.items():
            templates = by_suit[suit_code]
            if templates:
                print(f"\n{suit_name}:")
                for t in sorted(templates, key=lambda x: self.numbers.index(x['number'])):
                    print(f"  ✅ {t['full_name']}")
                    total += 1
        
        print(f"\n📊 Total: {total} plantillas")
        print(f"🎯 Objetivo: 52 cartas (baraja completa)")
        
        if total < 52:
            print(f"⚠️  Faltan {52 - total} cartas")


def main():
    """Menú principal"""
    capture = CardTemplateCapture()
    
    while True:
        print("\n" + "=" * 70)
        print("🃏 SISTEMA DE CAPTURA MEJORADO - ESQUINAS Y PALOS")
        print("=" * 70)
        print("1. 📸 Capturar plantillas (nuevo sistema)")
        print("2. 📋 Listar plantillas guardadas")
        print("3. 🗑️  Limpiar todas las plantillas")
        print("4. ❌ Salir")
        print("=" * 70)
        
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            camera = input("Índice de cámara (default=0): ").strip()
            camera_idx = int(camera) if camera else 0
            
            print("\n💡 CONSEJOS:")
            print("• Ten preparadas las 52 cartas")
            print("• Mantén buena iluminación")
            print("• Coloca las cartas bien verticales")
            print("• Una a la vez, sin prisa")
            
            input("\n⏸️  Presiona ENTER para continuar...")
            capture.capture_template_interactive(camera_idx)
        
        elif choice == '2':
            capture.list_templates()
        
        elif choice == '3':
            confirm = input("\n⚠️  ¿Eliminar TODAS las plantillas? (escribe 'SI'): ")
            if confirm.upper() == 'SI':
                capture.config = {'templates': [], 'corners': [], 'suits': []}
                capture.save_config()
                print("🗑️  Configuración limpiada")
                
                # Opcional: eliminar archivos
                delete_files = input("¿Eliminar archivos físicos también? (s/n): ")
                if delete_files.lower() == 's':
                    import shutil
                    try:
                        shutil.rmtree(capture.templates_dir)
                        os.makedirs(capture.templates_dir)
                        os.makedirs(capture.corners_dir)
                        os.makedirs(capture.suits_dir)
                        print("✅ Archivos eliminados")
                    except Exception as e:
                        print(f"❌ Error: {e}")
        
        elif choice == '4':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción no válida")


if __name__ == "__main__":
    main()