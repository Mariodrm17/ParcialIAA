import cv2
import threading
import time

class IVCamManager:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.available_cameras = []
        self.on_frame_callback = None
        self.lock = threading.Lock()  # 🔒 PARA THREAD-SAFETY
        
    def scan_cameras(self):
        """Escanea cámaras disponibles con mejor información"""
        self.available_cameras = []
        print("🔍 Escaneando cámaras...")
        
        # Probar índices 0, 1, 2, 3
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    # Dar tiempo para inicializar
                    time.sleep(0.3)
                    
                    # Intentar leer frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Intentar detectar el backend
                        backend = "Desconocido"
                        try:
                            backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))
                            backends = {
                                0: "Auto", 
                                200: "Media Foundation", 
                                1800: "Microsoft Media Foundation",
                                720897: "DirectShow",
                                1448695129: "V4L2"
                            }
                            backend = backends.get(backend_id, f"Backend {backend_id}")
                        except:
                            pass
                        
                        camera_info = {
                            'index': i,
                            'resolution': f"{width}x{height}",
                            'name': f"Cámara {i} ({backend})",
                            'backend': backend
                        }
                        
                        self.available_cameras.append(camera_info)
                        print(f"✅ {camera_info['name']} - {width}x{height}")
                    else:
                        print(f"⚠️  Cámara {i} abierta pero no produce video")
                    
                    cap.release()
                else:
                    print(f"❌ Cámara {i} no disponible")
                    
            except Exception as e:
                # Ignorar errores de cámaras no disponibles
                continue
                
        if not self.available_cameras:
            print("❌ No se encontraron cámaras disponibles")
        else:
            print(f"🎯 Total cámaras encontradas: {len(self.available_cameras)}")
                
        return self.available_cameras
    
    def start_camera(self, camera_index=0):
        """Inicia la cámara con mejor manejo de errores"""
        try:
            print(f"🚀 Iniciando cámara {camera_index}")
            
            # Verificar si la cámara existe en las disponibles
            available_indices = [cam['index'] for cam in self.available_cameras]
            if camera_index not in available_indices:
                return False, f"Cámara {camera_index} no encontrada. Primero ejecuta scan_cameras()"
            
            # Cerrar cámara anterior si existe
            if self.camera:
                self.camera.release()
                time.sleep(0.5)
            
            # Intentar abrir cámara con diferentes backends si es necesario
            backends_to_try = [
                cv2.CAP_DSHOW,  # DirectShow (Windows)
                cv2.CAP_MSMF,   # Media Foundation (Windows)
                cv2.CAP_ANY     # Auto-detect
            ]
            
            camera_opened = False
            for backend in backends_to_try:
                try:
                    self.camera = cv2.VideoCapture(camera_index, backend)
                    if self.camera.isOpened():
                        # Probar si realmente funciona
                        ret, test_frame = self.camera.read()
                        if ret and test_frame is not None:
                            camera_opened = True
                            print(f"✅ Cámara {camera_index} abierta con backend {backend}")
                            break
                        else:
                            self.camera.release()
                except Exception as e:
                    continue
            
            if not camera_opened:
                return False, f"No se pudo abrir la cámara {camera_index} con ningún backend"
            
            # Configuración óptima para cartas
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            # Esperar inicialización
            time.sleep(1)
            
            # Verificar que funciona correctamente
            working_frames = 0
            for _ in range(10):
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    working_frames += 1
                time.sleep(0.1)
            
            if working_frames == 0:
                self.camera.release()
                return False, "Cámara abierta pero no produce video"
            
            self.is_running = True
            
            # Iniciar hilo de captura
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            # Obtener configuración real
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            success_msg = f"Cámara {camera_index} iniciada - {actual_width}x{actual_height} @ {actual_fps:.1f}FPS"
            print(f"✅ {success_msg}")
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Error crítico: {str(e)}"
            print(f"❌ {error_msg}")
            if self.camera:
                self.camera.release()
                self.camera = None
            return False, error_msg
    
    def _capture_loop(self):
        """Loop de captura mejorado con manejo de errores"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running and self.camera:
            try:
                with self.lock:
                    ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    self.current_frame = frame
                    consecutive_errors = 0  # Resetear contador de errores
                    
                    # Llamar callback si existe
                    if self.on_frame_callback:
                        try:
                            self.on_frame_callback(frame)
                        except Exception as e:
                            print(f"Error en callback: {e}")
                
                else:
                    consecutive_errors += 1
                    print(f"⚠️  Error leyendo frame ({consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("❌ Demasiados errores consecutivos, deteniendo cámara")
                        self.stop_camera()
                        break
                
                # Control de FPS (no usar time.sleep() aquí para mejor responsividad)
                # El control real de FPS lo hace OpenCV internamente
                
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ Error en captura: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("❌ Demasiados errores, deteniendo cámara")
                    self.stop_camera()
                    break
                
                time.sleep(0.1)  # Pequeña pausa antes de reintentar
    
    def stop_camera(self):
        """Detiene la cámara de forma segura"""
        print("🛑 Deteniendo cámara...")
        self.is_running = False
        
        # Esperar a que el hilo termine
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Liberar cámara
        if self.camera:
            with self.lock:
                self.camera.release()
            self.camera = None
        
        self.current_frame = None
        print("✅ Cámara detenida correctamente")
    
    def get_frame(self):
        """Obtiene el frame actual de forma segura"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def set_frame_callback(self, callback):
        """Establece callback para nuevos frames"""
        self.on_frame_callback = callback
    
    def is_camera_working(self):
        """Verifica si la cámara está funcionando"""
        return (self.is_running and 
                self.camera is not None and 
                self.current_frame is not None)
    
    def get_camera_info(self):
        """Obtiene información de la cámara actual"""
        if not self.camera:
            return None
            
        try:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            backend = int(self.camera.get(cv2.CAP_PROP_BACKEND))
            
            return {
                'resolution': f"{width}x{height}",
                'fps': fps,
                'backend': backend,
                'is_running': self.is_running
            }
        except:
            return None
    
    def restart_camera(self, camera_index=None):
        """Reinicia la cámara (útil para recuperación)"""
        print("🔄 Reiniciando cámara...")
        
        current_index = 0
        if self.camera:
            # Intentar obtener el índice actual
            try:
                current_index = int(self.camera.get(cv2.CAP_PROP_POS_FRAMES))
            except:
                pass
        
        # Usar nuevo índice si se proporciona, sino el actual
        target_index = camera_index if camera_index is not None else current_index
        
        self.stop_camera()
        time.sleep(1)  # Dar tiempo para liberar recursos
        
        return self.start_camera(target_index)

# Función de prueba
def test_camera_manager():
    """Prueba el camera manager"""
    print("🧪 Probando Camera Manager...")
    
    manager = IVCamManager()
    
    # Escanear cámaras
    cameras = manager.scan_cameras()
    
    if not cameras:
        print("❌ No hay cámaras para probar")
        return
    
    # Usar primera cámara disponible
    cam_index = cameras[0]['index']
    print(f"🎥 Probando con cámara {cam_index}")
    
    # Iniciar cámara
    success, message = manager.start_camera(cam_index)
    
    if success:
        print(f"✅ {message}")
        
        # Mostrar video por 5 segundos
        start_time = time.time()
        while time.time() - start_time < 5:
            frame = manager.get_frame()
            if frame is not None:
                # Mostrar información en el frame
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Camara {cam_index} - Prueba", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Presiona 'q' para salir", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Prueba Camera Manager", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Detener cámara
        manager.stop_camera()
        cv2.destroyAllWindows()
        print("✅ Prueba completada")
    else:
        print(f"❌ Error: {message}")

if __name__ == "__main__":
    test_camera_manager()