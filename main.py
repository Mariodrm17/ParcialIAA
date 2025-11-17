import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import time
import threading
from datetime import datetime

# Importar los módulos actualizados
from camera_manager import IVCamManager
from poker_card_detector import PokerCardDetector  # ✅ NUEVO DETECTOR
from number_recognizer import NumberRecognizer
from suit_recognizer import SuitRecognizer

class PokerCardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Cartas Poker - Sistema Mejorado")
        self.root.geometry("1200x800")
        
        # Inicializar componentes ACTUALIZADOS
        self.camera_manager = IVCamManager()
        self.card_detector = PokerCardDetector()  # ✅ NUEVO DETECTOR
        self.number_recognizer = NumberRecognizer()
        self.suit_recognizer = SuitRecognizer()
        
        # Estado
        self.is_running = False
        self.current_frame = None
        self.detected_cards = []
        self.last_process_time = 0
        self.calibrated = False
        
        self.create_interface()
        self.scan_cameras()
        self.update_display()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_interface(self):
        """Interfaz mejorada"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control
        control_frame = ttk.LabelFrame(main_frame, text="Control del Sistema", padding="10")
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
        
        # Botones de control MEJORADOS
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(btn_frame, text="🎥 Iniciar Cámara", 
                                   command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Detener", 
                                  command=self.stop_camera, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.calibrate_btn = ttk.Button(btn_frame, text="🎨 Calibrar Tapete", 
                                       command=self.calibrate_detector)
        self.calibrate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(btn_frame, text="💾 Guardar", 
                  command=self.save_results).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(btn_frame, text="🧹 Limpiar", 
                  command=self.clear_detections).pack(side=tk.LEFT)
        
        # Contenido principal
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Vista de cámara
        cam_frame = ttk.LabelFrame(content_frame, text="Vista en Vivo")
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.cam_label = ttk.Label(cam_frame, 
                                  text="Sistema de Detección de Cartas\n\n"
                                       "1. Haz clic en 'Buscar Cámaras'\n"
                                       "2. Selecciona una cámara\n"
                                       "3. Haz clic en 'Iniciar Cámara'\n"
                                       "4. CALIBRA con 'Calibrar Tapete'\n"
                                       "5. Coloca cartas sobre el tapete verde\n\n"
                                       "⚠️  CALIBRACIÓN ES ESENCIAL",
                                  background="black", foreground="white",
                                  justify=tk.CENTER, font=("Arial", 11))
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(content_frame, text="Cartas Detectadas")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Lista de cartas
        list_container = ttk.Frame(results_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cards_list = tk.Listbox(list_container, font=("Arial", 11), height=20)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.cards_list.yview)
        self.cards_list.configure(yscrollcommand=scrollbar.set)
        
        self.cards_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Estado
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Listo para iniciar")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # Indicador de calibración
        self.calibration_status = tk.StringVar(value="❌ NO CALIBRADO")
        ttk.Label(status_frame, textvariable=self.calibration_status, 
                 foreground="red").pack(side=tk.RIGHT)
    
    def scan_cameras(self):
        """Escanea cámaras disponibles"""
        self.status_var.set("Buscando cámaras...")
        cameras = self.camera_manager.scan_cameras()
        
        cam_list = [f"{cam['name']} (Índice {cam['index']})" for cam in cameras]
        self.cam_combo['values'] = cam_list
        
        if cam_list:
            self.cam_combo.current(0)
            self.status_var.set(f"Encontradas {len(cameras)} cámaras")
        else:
            self.status_var.set("No se encontraron cámaras")
    
    def calibrate_detector(self):
        """Calibra el detector de cartas"""
        if not self.is_running:
            messagebox.showwarning("Advertencia", "Primero inicia la cámara para calibrar")
            return
        
        if self.current_frame is not None:
            self.status_var.set("Iniciando calibración...")
            success = self.card_detector.calibrate_green_color(self.current_frame)
            
            if success:
                self.calibrated = True
                self.calibration_status.set("✅ CALIBRADO")
                self.status_var.set("Sistema calibrado correctamente")
                messagebox.showinfo("Éxito", "Calibración completada!\nAhora coloca cartas sobre el tapete.")
            else:
                self.status_var.set("Calibración cancelada")
        else:
            messagebox.showerror("Error", "No hay imagen de cámara para calibrar")
    
    def start_camera(self):
        """Inicia la cámara seleccionada"""
        if not self.cam_combo['values']:
            messagebox.showerror("Error", "Primero busca las cámaras disponibles")
            return
        
        try:
            selected = self.cam_var.get()
            if not selected:
                self.cam_combo.current(0)
                selected = self.cam_combo.get()
            
            # Extraer índice de cámara
            cam_index = int(selected.split("Índice ")[1].split(")")[0])
            
            # Configurar callback
            self.camera_manager.set_frame_callback(self.on_camera_frame)
            
            # Iniciar cámara
            self.status_var.set(f"Iniciando cámara {cam_index}...")
            success, message = self.camera_manager.start_camera(cam_index)
            
            if success:
                self.is_running = True
                self.status_var.set(message)
                self.start_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
                self.cam_label.configure(text="Cámara iniciada\n\nAhora CALIBRA el tapete")
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
            text="Cámara detenida\n\nHaz clic en 'Iniciar Cámara'",
            background="black", 
            foreground="white"
        )
        self.status_var.set("Cámara detenida")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.calibrated = False
        self.calibration_status.set("❌ NO CALIBRADO")
    
    def on_camera_frame(self, frame):
        """Procesa frame de la cámara"""
        self.current_frame = frame
        
        # Procesar detección de cartas (cada 1.5 segundos si está calibrado)
        current_time = time.time()
        if self.calibrated and (current_time - self.last_process_time >= 1.5):
            self.last_process_time = current_time
            self.process_frame(frame)
    
    def process_frame(self, frame):
        """Procesa frame para detectar cartas - ACTUALIZADO"""
        try:
            # Usar el NUEVO método de detección
            card_images, card_positions = self.card_detector.detect_cards(frame)
            
            # Procesar cada carta detectada
            for card_img, position in zip(card_images, card_positions):
                try:
                    # Preprocesar la carta para mejor reconocimiento
                    processed_card = self.preprocess_card_image(card_img)
                    
                    # Reconocer número y palo
                    number = self.number_recognizer.recognize(processed_card)
                    suit_code = self.suit_recognizer.recognize(processed_card)
                    suit_name = self.suit_recognizer.get_suit_name(suit_code)
                    
                    if number and suit_name:
                        card_name = f"{number} de {suit_name}"
                        if card_name not in self.detected_cards:
                            self.detected_cards.append(card_name)
                            self.status_var.set(f"Detectada: {card_name}")
                            print(f"🎴 Nueva carta detectada: {card_name}")
                            
                except Exception as e:
                    print(f"Error procesando carta individual: {e}")
                    continue
                        
        except Exception as e:
            print(f"Error en process_frame: {e}")
    
    def preprocess_card_image(self, card_image):
        """Preprocesa la imagen de la carta para mejor reconocimiento"""
        try:
            # Redimensionar a tamaño estándar
            card_std = cv2.resize(card_image, (200, 300))
            
            # Mejorar contraste
            lab = cv2.cvtColor(card_std, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0).apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            print(f"Error en preprocesamiento de carta: {e}")
            return card_image
    
    def update_display(self):
        """Actualiza la pantalla"""
        try:
            if self.is_running and self.current_frame is not None:
                # Crear frame para mostrar
                display_frame = self.current_frame.copy()
                
                # Si está calibrado, mostrar detecciones
                if self.calibrated:
                    card_images, card_positions = self.card_detector.detect_cards(self.current_frame)
                    
                    # Dibujar rectángulos alrededor de las cartas
                    for i, (x, y, w, h) in enumerate(card_positions):
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(display_frame, f"Carta {i+1}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Redimensionar para mostrar
                display_frame = cv2.resize(display_frame, (800, 600))
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.cam_label.imgtk = imgtk
                self.cam_label.configure(image=imgtk)
            
            # Actualizar lista de cartas
            self.cards_list.delete(0, tk.END)
            for card in self.detected_cards:
                self.cards_list.insert(tk.END, card)
            
            # Siguiente actualización
            self.root.after(50, self.update_display)
            
        except Exception as e:
            print(f"Error en display: {e}")
            self.root.after(100, self.update_display)
    
    def save_results(self):
        """Guarda los resultados"""
        if not self.detected_cards:
            messagebox.showinfo("Información", "No hay cartas detectadas")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cartas_detectadas_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("CARTAS DETECTADAS - SISTEMA MEJORADO\n")
                f.write("=" * 40 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total cartas: {len(self.detected_cards)}\n\n")
                
                for i, card in enumerate(self.detected_cards, 1):
                    f.write(f"{i}. {card}\n")
            
            messagebox.showinfo("Éxito", f"Resultados guardados en:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {e}")
    
    def clear_detections(self):
        """Limpia las detecciones"""
        self.detected_cards = []
        self.cards_list.delete(0, tk.END)
        self.status_var.set("Detecciones limpiadas")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        self.camera_manager.stop_camera()
        self.root.destroy()

def main():
    # Configuración para evitar warnings
    import os
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    root = tk.Tk()
    app = PokerCardApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()