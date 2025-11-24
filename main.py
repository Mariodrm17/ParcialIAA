import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import time
import json
from datetime import datetime

# ✅ IMPORTAR MÓDULOS CORRECTOS
from camera_manager import IVCamManager
from fixed_template_matcher import FixedTemplateRecognizer


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
                            margin = 20
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
        self.root.title("🃏 Detector de Cartas Poker - Reconocimiento por Esquinas")
        self.root.geometry("1400x900")
        
        # ✅ COMPONENTES CORRECTOS
        self.camera_manager = IVCamManager()
        self.card_detector = CardDetectorWithCalibration()
        self.recognizer = FixedTemplateRecognizer()  # ✅ El que reconoce esquinas
        
        # Estado
        self.is_running = False
        self.current_frame = None
        self.detected_cards = set()  # ✅ Usar set para evitar duplicados
        self.last_process_time = 0
        self.process_interval = 1.0  # Procesar cada 1 segundo
        
        self.create_interface()
        self.scan_cameras()
        self.update_display()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_interface(self):
        """Interfaz mejorada"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control superior
        control_frame = ttk.LabelFrame(main_frame, text="🎮 Control del Sistema", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Fila 1: Selección de cámara
        select_frame = ttk.Frame(control_frame)
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(select_frame, text="🔍 Buscar Cámaras", 
                  command=self.scan_cameras, width=15).pack(side=tk.LEFT, padx=(0, 10))
        
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(select_frame, textvariable=self.cam_var,
                                     state="readonly", width=50)
        self.cam_combo.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        # Fila 2: Botones de control
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(btn_frame, text="▶️ Iniciar", 
                                   command=self.start_camera, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Detener", 
                                  command=self.stop_camera, state="disabled", width=12)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(btn_frame, text="🔄 Recalibrar", 
                  command=self.reload_calibration, width=12).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(btn_frame, text="💾 Guardar", 
                  command=self.save_results, width=12).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(btn_frame, text="🧹 Limpiar", 
                  command=self.clear_detections, width=12).pack(side=tk.LEFT)
        
        # Contenido principal - 2 columnas
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Columna izquierda: Cámara
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        cam_label_frame = ttk.LabelFrame(left_frame, text="📹 Vista en Vivo")
        cam_label_frame.pack(fill=tk.BOTH, expand=True)
        
        # Estado de calibración
        calibration_status = "✅ CALIBRADO" if self.card_detector.is_calibrated else "❌ NO CALIBRADO"
        templates_status = f"📦 {len(self.recognizer.corner_templates)} plantillas" if self.recognizer.corner_templates else "❌ SIN PLANTILLAS"
        
        self.cam_label = ttk.Label(cam_label_frame, 
                                  text=f"🃏 Sistema de Detección de Cartas\n\n"
                                       f"Calibración: {calibration_status}\n"
                                       f"Plantillas: {templates_status}\n\n"
                                       "📋 PASOS:\n"
                                       "1. Buscar Cámaras\n"
                                       "2. Seleccionar cámara\n"
                                       "3. Iniciar Cámara\n"
                                       "4. Colocar cartas\n\n"
                                       "⚠️  Si no funciona:\n"
                                       "• python green_calibrator.py (tapete)\n"
                                       "• python template_capture.py (cartas)",
                                  background="black", foreground="white",
                                  justify=tk.CENTER, font=("Courier", 10))
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Columna derecha: Resultados
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        results_frame = ttk.LabelFrame(right_frame, text="🎴 Cartas Detectadas")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Lista con scrollbar
        list_container = ttk.Frame(results_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cards_list = tk.Listbox(list_container, font=("Courier", 12), 
                                     width=30, height=25)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", 
                                 command=self.cards_list.yview)
        self.cards_list.configure(yscrollcommand=scrollbar.set)
        
        self.cards_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Estadísticas
        stats_frame = ttk.LabelFrame(right_frame, text="📊 Estadísticas", padding="5")
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_var = tk.StringVar(value="Total: 0 cartas")
        ttk.Label(stats_frame, textvariable=self.stats_var, 
                 font=("Courier", 10)).pack()
        
        # Barra de estado
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        calibration_text = "✅ Sistema calibrado" if self.card_detector.is_calibrated else "❌ NO calibrado - ejecuta green_calibrator.py"
        calibration_color = "green" if self.card_detector.is_calibrated else "red"
        
        self.status_var = tk.StringVar(value=calibration_text)
        ttk.Label(status_frame, textvariable=self.status_var, 
                 foreground=calibration_color, font=("Arial", 9)).pack(side=tk.LEFT)
    
    def scan_cameras(self):
        """Escanea cámaras disponibles"""
        self.status_var.set("🔍 Buscando cámaras...")
        self.root.update()
        
        cameras = self.camera_manager.scan_cameras()
        
        cam_list = [f"{cam['name']} - {cam['resolution']}" for cam in cameras]
        self.cam_combo['values'] = cam_list
        
        if cam_list:
            self.cam_combo.current(0)
            self.status_var.set(f"✅ {len(cameras)} cámaras encontradas")
        else:
            self.status_var.set("❌ No se encontraron cámaras")
            messagebox.showwarning("Advertencia", "No se encontraron cámaras disponibles")
    
    def reload_calibration(self):
        """Recarga calibración y plantillas"""
        # Recargar calibración del tapete
        calib_success = self.card_detector.load_calibration()
        
        # Recargar plantillas
        template_success = self.recognizer.load_templates()
        
        if calib_success and template_success:
            self.status_var.set("✅ Calibración y plantillas recargadas")
            messagebox.showinfo("Éxito", 
                              f"✅ Sistema recargado\n"
                              f"Calibración: OK\n"
                              f"Plantillas: {len(self.recognizer.corner_templates)}")
        elif calib_success:
            self.status_var.set("⚠️  Calibración OK pero sin plantillas")
            messagebox.showwarning("Advertencia", 
                                 "Calibración OK pero no hay plantillas.\n"
                                 "Ejecuta: python template_capture.py")
        else:
            self.status_var.set("❌ Error al recargar")
            messagebox.showerror("Error", 
                               "No se pudo recargar la configuración.\n"
                               "Verifica los archivos de calibración.")
    
    def start_camera(self):
        """Inicia la cámara"""
        # Verificar sistema
        if not self.card_detector.is_calibrated:
            messagebox.showwarning("Sin Calibración", 
                                 "⚠️  NO HAY CALIBRACIÓN DEL TAPETE\n\n"
                                 "Ejecuta: python green_calibrator.py")
            return
        
        if not self.recognizer.corner_templates:
            messagebox.showwarning("Sin Plantillas", 
                                 "⚠️  NO HAY PLANTILLAS DE CARTAS\n\n"
                                 "Ejecuta: python template_capture.py")
            return
        
        if not self.cam_combo['values']:
            messagebox.showerror("Error", "Busca las cámaras primero")
            return
        
        try:
            selected = self.cam_var.get()
            if not selected:
                messagebox.showerror("Error", "Selecciona una cámara")
                return
            
            # Extraer índice de la cámara
            cam_index = None
            for cam in self.camera_manager.available_cameras:
                if cam['name'] in selected:
                    cam_index = cam['index']
                    break
            
            if cam_index is None:
                messagebox.showerror("Error", "No se pudo determinar el índice de la cámara")
                return
            
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
                print(f"\n✅ Sistema listo para detectar cartas")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar: {str(e)}")
    
    def stop_camera(self):
        """Detiene la cámara"""
        self.camera_manager.stop_camera()
        self.is_running = False
        self.cam_label.config(image='')
        self.cam_label.configure(
            text="⏹️ Cámara detenida\n\nClick 'Iniciar' para continuar",
            background="black", 
            foreground="white"
        )
        self.status_var.set("⏹️ Cámara detenida")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def on_camera_frame(self, frame):
        """Callback cuando llega un nuevo frame"""
        self.current_frame = frame
        
        # Procesar cada cierto intervalo
        current_time = time.time()
        if (current_time - self.last_process_time >= self.process_interval):
            self.last_process_time = current_time
            self.process_frame(frame)
    
    def process_frame(self, frame):
        """Procesa frame para detectar y reconocer cartas"""
        try:
            # Detectar cartas en el frame
            card_images, card_positions = self.card_detector.detect_cards(frame)
            
            # Reconocer cada carta detectada
            for card_img in card_images:
                try:
                    # ✅ RECONOCER CON ROTACIONES (más robusto)
                    card_name, confidence = self.recognizer.recognize_with_rotation(card_img)
                    
                    # Solo agregar si la confianza es suficiente
                    if card_name and confidence >= 0.6:
                        if card_name not in self.detected_cards:
                            self.detected_cards.add(card_name)
                            print(f"🎴 Nueva carta: {card_name} (conf: {confidence:.2%})")
                            
                except Exception as e:
                    print(f"Error reconociendo carta: {e}")
                    continue
                        
        except Exception as e:
            print(f"Error en process_frame: {e}")
    
    def update_display(self):
        """Actualiza la pantalla"""
        try:
            if self.is_running and self.current_frame is not None:
                display_frame = self.current_frame.copy()
                
                # Detectar cartas para visualización
                card_images, card_positions = self.card_detector.detect_cards(self.current_frame)
                
                # Dibujar detecciones
                for i, (x, y, w, h) in enumerate(card_positions):
                    # Intentar reconocer para mostrar
                    card_img = self.current_frame[y:y+h, x:x+w]
                    card_name, confidence = self.recognizer.recognize_card(card_img)
                    
                    # Color según confianza
                    if confidence >= 0.7:
                        color = (0, 255, 0)  # Verde
                    elif confidence >= 0.5:
                        color = (0, 255, 255)  # Amarillo
                    else:
                        color = (0, 0, 255)  # Rojo
                    
                    # Dibujar rectángulo
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Mostrar nombre si se reconoció
                    if card_name:
                        cv2.putText(display_frame, card_name, (x, y-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(display_frame, f"{confidence:.1%}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        cv2.putText(display_frame, f"? ({confidence:.1%})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Info general
                cv2.putText(display_frame, f"Cartas visibles: {len(card_images)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Reconocidas: {len(self.detected_cards)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Redimensionar para mostrar
                h, w = display_frame.shape[:2]
                display_w = 900
                display_h = int(h * display_w / w)
                display_frame = cv2.resize(display_frame, (display_w, display_h))
                
                # Convertir y mostrar
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.cam_label.imgtk = imgtk
                self.cam_label.configure(image=imgtk)
            
            # Actualizar lista de cartas
            self.cards_list.delete(0, tk.END)
            for i, card in enumerate(sorted(self.detected_cards), 1):
                self.cards_list.insert(tk.END, f"{i:2d}. {card}")
            
            # Actualizar estadísticas
            self.stats_var.set(f"Total: {len(self.detected_cards)} cartas")
            
            # Siguiente actualización
            self.root.after(30, self.update_display)
            
        except Exception as e:
            print(f"Error en display: {e}")
            self.root.after(100, self.update_display)
    
    def save_results(self):
        """Guarda resultados en archivo"""
        if not self.detected_cards:
            messagebox.showinfo("Info", "No hay cartas para guardar")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cartas_detectadas_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("🃏 CARTAS DE POKER DETECTADAS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total: {len(self.detected_cards)} cartas\n")
                f.write("=" * 50 + "\n\n")
                
                for i, card in enumerate(sorted(self.detected_cards), 1):
                    f.write(f"{i:2d}. {card}\n")
                
                f.write("\n" + "=" * 50 + "\n")
            
            messagebox.showinfo("Guardado", f"✅ Archivo guardado:\n{filename}")
            print(f"💾 Resultados guardados en: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar: {e}")
    
    def clear_detections(self):
        """Limpia lista de detecciones"""
        self.detected_cards.clear()
        self.cards_list.delete(0, tk.END)
        self.stats_var.set("Total: 0 cartas")
        self.status_var.set("🧹 Detecciones limpiadas")
        print("🧹 Lista de cartas limpiada")
    
    def on_closing(self):
        """Cierre limpio de la aplicación"""
        print("Cerrando aplicación...")
        self.camera_manager.stop_camera()
        self.root.destroy()


def main():
    # Configuración
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    print("=" * 70)
    print("🃏 DETECTOR DE CARTAS POKER")
    print("   Reconocimiento por Template Matching (Esquinas)")
    print("=" * 70)
    print("\n📋 REQUISITOS:")
    print("✓ Calibración del tapete verde (green_calibration.json)")
    print("✓ Plantillas de cartas (templates_config.json)")
    print("\n💡 Si faltan archivos, ejecuta:")
    print("   1. python green_calibrator.py")
    print("   2. python template_capture.py")
    print("=" * 70 + "\n")
    
    # Verificar archivos necesarios
    missing_files = []
    if not os.path.exists("green_calibration.json"):
        missing_files.append("green_calibration.json")
    if not os.path.exists("templates_config.json"):
        missing_files.append("templates_config.json")
    
    if missing_files:
        print("⚠️  ADVERTENCIA: Faltan archivos:")
        for f in missing_files:
            print(f"   ❌ {f}")
        print("\nLa aplicación puede no funcionar correctamente.")
        input("\nPresiona ENTER para continuar de todos modos...")
    
    # Iniciar aplicación
    root = tk.Tk()
    app = PokerCardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()