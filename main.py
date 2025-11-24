import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import time
import json
from datetime import datetime

# Importar módulos correctos
from camera_manager import IVCamManager
from number_recognizer import NumberRecognizer
from suit_recognizer import SuitRecognizer

class CardDetectorWithCalibration:
    """Detector que USA la calibración guardada"""
    
    def __init__(self):
        self.green_lower = None
        self.green_upper = None
        self.is_calibrated = False
        self.min_area = 2000
        self.max_area = 50000
        
        # Intentar cargar calibración automáticamente
        self.load_calibration()
    
    def load_calibration(self):
        """Carga la calibración del archivo JSON"""
        config_file = "green_calibration.json"
        
        if not os.path.exists(config_file):
            print("⚠️  No se encontró calibración. Ejecuta green_calibrator.py primero")
            return False
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.green_lower = np.array(config['green_lower'])
            self.green_upper = np.array(config['green_upper'])
            self.is_calibrated = True
            
            print("✅ Calibración cargada exitosamente")
            print(f"   Lower: {self.green_lower}")
            print(f"   Upper: {self.green_upper}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando calibración: {e}")
            return False
    
    def detect_cards(self, frame):
        """Detecta cartas usando la calibración"""
        if not self.is_calibrated:
            print("⚠️  No hay calibración cargada")
            return [], []
        
        try:
            # Convertir a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Crear máscara verde
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Limpiar máscara
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            
            # Invertir máscara (cartas = blanco)
            cards_mask = cv2.bitwise_not(green_mask)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(cards_mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
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
                        
                        # Verificar relación de aspecto
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.5 <= aspect_ratio <= 0.9:
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


class PokerCardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🃏 Detector de Cartas Poker - CON CALIBRACIÓN")
        self.root.geometry("1200x800")
        
        # Inicializar componentes CORRECTOS
        self.camera_manager = IVCamManager()
        self.card_detector = CardDetectorWithCalibration()  # ✅ DETECTOR CORRECTO
        self.number_recognizer = NumberRecognizer()
        self.suit_recognizer = SuitRecognizer()
        
        # Estado
        self.is_running = False
        self.current_frame = None
        self.detected_cards = []
        self.last_process_time = 0
        
        self.create_interface()
        self.scan_cameras()
        self.update_display()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_interface(self):
        """Interfaz simplificada"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control
        control_frame = ttk.LabelFrame(main_frame, text="🎮 Control del Sistema", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Selección de cámara
        select_frame = ttk.Frame(control_frame)
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(select_frame, text="🔍 Buscar Cámaras", 
                  command=self.scan_cameras).pack(side=tk.LEFT, padx=(0, 10))
        
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(select_frame, textvariable=self.cam_var,
                                     state="readonly", width=40)
        self.cam_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botones de control
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(btn_frame, text="▶️ Iniciar Cámara", 
                                   command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Detener", 
                                  command=self.stop_camera, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(btn_frame, text="🔄 Recargar Calibración", 
                  command=self.reload_calibration).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(btn_frame, text="💾 Guardar", 
                  command=self.save_results).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(btn_frame, text="🧹 Limpiar", 
                  command=self.clear_detections).pack(side=tk.LEFT)
        
        # Contenido principal
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Vista de cámara
        cam_frame = ttk.LabelFrame(content_frame, text="📹 Vista en Vivo")
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        calibration_status = "✅ CALIBRADO" if self.card_detector.is_calibrated else "❌ NO CALIBRADO"
        
        self.cam_label = ttk.Label(cam_frame, 
                                  text=f"🃏 Sistema de Detección de Cartas\n\n"
                                       f"Estado: {calibration_status}\n\n"
                                       "📋 PASOS:\n"
                                       "1. Buscar Cámaras\n"
                                       "2. Seleccionar cámara\n"
                                       "3. Iniciar Cámara\n"
                                       "4. Colocar cartas sobre tapete verde\n\n"
                                       "⚠️  Si no detecta, ejecuta:\n"
                                       "   python green_calibrator.py",
                                  background="black", foreground="white",
                                  justify=tk.CENTER, font=("Arial", 11))
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(content_frame, text="🎴 Cartas Detectadas")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Lista de cartas
        list_container = ttk.Frame(results_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cards_list = tk.Listbox(list_container, font=("Arial", 11), height=20)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", 
                                 command=self.cards_list.yview)
        self.cards_list.configure(yscrollcommand=scrollbar.set)
        
        self.cards_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Estado
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        calibration_text = "✅ Sistema calibrado" if self.card_detector.is_calibrated else "❌ NO calibrado - ejecuta green_calibrator.py"
        calibration_color = "green" if self.card_detector.is_calibrated else "red"
        
        self.status_var = tk.StringVar(value=calibration_text)
        ttk.Label(status_frame, textvariable=self.status_var, 
                 foreground=calibration_color).pack(side=tk.LEFT)
    
    def scan_cameras(self):
        """Escanea cámaras disponibles"""
        self.status_var.set("🔍 Buscando cámaras...")
        cameras = self.camera_manager.scan_cameras()
        
        cam_list = [f"{cam['name']} (Índice {cam['index']})" for cam in cameras]
        self.cam_combo['values'] = cam_list
        
        if cam_list:
            self.cam_combo.current(0)
            self.status_var.set(f"✅ Encontradas {len(cameras)} cámaras")
        else:
            self.status_var.set("❌ No se encontraron cámaras")
    
    def reload_calibration(self):
        """Recarga la calibración"""
        success = self.card_detector.load_calibration()
        
        if success:
            self.status_var.set("✅ Calibración recargada exitosamente")
            messagebox.showinfo("Éxito", "Calibración recargada correctamente")
        else:
            self.status_var.set("❌ Error al recargar calibración")
            messagebox.showerror("Error", 
                               "No se pudo cargar la calibración.\n"
                               "Ejecuta: python green_calibrator.py")
    
    def start_camera(self):
        """Inicia la cámara seleccionada"""
        if not self.card_detector.is_calibrated:
            messagebox.showwarning("Advertencia", 
                                 "⚠️  NO HAY CALIBRACIÓN\n\n"
                                 "Ejecuta primero:\n"
                                 "python green_calibrator.py")
            return
        
        if not self.cam_combo['values']:
            messagebox.showerror("Error", "Primero busca las cámaras disponibles")
            return
        
        try:
            selected = self.cam_var.get()
            if not selected:
                self.cam_combo.current(0)
                selected = self.cam_combo.get()
            
            # Extraer índice
            cam_index = int(selected.split("Índice ")[1].split(")")[0])
            
            # Configurar callback
            self.camera_manager.set_frame_callback(self.on_camera_frame)
            
            # Iniciar cámara
            self.status_var.set(f"🚀 Iniciando cámara {cam_index}...")
            success, message = self.camera_manager.start_camera(cam_index)
            
            if success:
                self.is_running = True
                self.status_var.set(f"✅ {message}")
                self.start_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar cámara: {str(e)}")
    
    def stop_camera(self):
        """Detiene la cámara"""
        self.camera_manager.stop_camera()
        self.is_running = False
        self.cam_label.config(image='')
        self.cam_label.configure(
            text="⏹️ Cámara detenida\n\nHaz clic en 'Iniciar Cámara'",
            background="black", 
            foreground="white"
        )
        self.status_var.set("⏹️ Cámara detenida")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def on_camera_frame(self, frame):
        """Procesa frame de la cámara"""
        self.current_frame = frame
        
        # Procesar detección cada 1.5 segundos
        current_time = time.time()
        if (current_time - self.last_process_time >= 1.5):
            self.last_process_time = current_time
            self.process_frame(frame)
    
    def process_frame(self, frame):
        """Procesa frame para detectar cartas"""
        try:
            # Detectar cartas
            card_images, card_positions = self.card_detector.detect_cards(frame)
            
            # Procesar cada carta
            for card_img, position in zip(card_images, card_positions):
                try:
                    # Preprocesar
                    processed_card = self.preprocess_card_image(card_img)
                    
                    # Reconocer
                    number = self.number_recognizer.recognize(processed_card)
                    suit_code = self.suit_recognizer.recognize(processed_card)
                    suit_name = self.suit_recognizer.get_suit_name(suit_code)
                    
                    if number and suit_name:
                        card_name = f"{number} de {suit_name}"
                        if card_name not in self.detected_cards:
                            self.detected_cards.append(card_name)
                            self.status_var.set(f"🎴 Detectada: {card_name}")
                            print(f"🃏 Nueva carta: {card_name}")
                            
                except Exception as e:
                    print(f"Error procesando carta: {e}")
                    continue
                        
        except Exception as e:
            print(f"Error en process_frame: {e}")
    
    def preprocess_card_image(self, card_image):
        """Preprocesa imagen de carta"""
        try:
            card_std = cv2.resize(card_image, (200, 300))
            
            lab = cv2.cvtColor(card_std, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0).apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return card_image
    
    def update_display(self):
        """Actualiza la pantalla"""
        try:
            if self.is_running and self.current_frame is not None:
                display_frame = self.current_frame.copy()
                
                # Detectar y mostrar cartas
                card_images, card_positions = self.card_detector.detect_cards(self.current_frame)
                
                # Dibujar rectángulos
                for i, (x, y, w, h) in enumerate(card_positions):
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(display_frame, f"Carta {i+1}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Información
                cv2.putText(display_frame, f"Cartas: {len(card_images)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Redimensionar
                display_frame = cv2.resize(display_frame, (800, 600))
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.cam_label.imgtk = imgtk
                self.cam_label.configure(image=imgtk)
            
            # Actualizar lista
            self.cards_list.delete(0, tk.END)
            for card in self.detected_cards:
                self.cards_list.insert(tk.END, card)
            
            # Siguiente actualización
            self.root.after(50, self.update_display)
            
        except Exception as e:
            print(f"Error en display: {e}")
            self.root.after(100, self.update_display)
    
    def save_results(self):
        """Guarda resultados"""
        if not self.detected_cards:
            messagebox.showinfo("Información", "No hay cartas detectadas")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cartas_detectadas_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🃏 CARTAS DETECTADAS\n")
                f.write("=" * 40 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total: {len(self.detected_cards)}\n\n")
                
                for i, card in enumerate(self.detected_cards, 1):
                    f.write(f"{i}. {card}\n")
            
            messagebox.showinfo("Éxito", f"Guardado en:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar: {e}")
    
    def clear_detections(self):
        """Limpia detecciones"""
        self.detected_cards = []
        self.cards_list.delete(0, tk.END)
        self.status_var.set("🧹 Detecciones limpiadas")
    
    def on_closing(self):
        """Cierre limpio"""
        self.camera_manager.stop_camera()
        self.root.destroy()


def main():
    # Suprimir warnings
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    print("=" * 60)
    print("🃏 DETECTOR DE CARTAS POKER")
    print("=" * 60)
    print("\n⚠️  IMPORTANTE:")
    print("Si es la primera vez, ejecuta primero:")
    print("   python green_calibrator.py")
    print("\nPara calibrar el tapete verde.")
    print("=" * 60)
    
    root = tk.Tk()
    app = PokerCardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()