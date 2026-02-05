import os
import time
import subprocess
import zmq
import sys

# CONFIGURAZIONE HARDCODED (Come richiesto)
RCT_EXECUTABLE = "/home/francesco/ML/OpenRCT2-fork/build/openrct2"
RCT_DATA_PATH = "/home/francesco/ML/RCT2_Assets/Rollercoaster Tycoon 2"
# Percorso di default per i salvataggi/scenari se necessario
BASE_PATH = "/home/francesco/ML/rctrl/"


class Bridge:
    def __init__(self, port, headless=False):
        self.context = zmq.Context()
        self.bound = False
        self.port = str(port)
        self.headless = headless
        self.rct_process = None

    def bind(self):

        os.system(f"fuser -k {self.port}/tcp {int(self.port)+1}/tcp 2>/dev/null || true")
        
        print(f"[*] Binding ZMQ (Client/Server) su 127.0.0.1 ports {self.port} / {int(self.port)+1}")

        # FIX RETE: Usiamo 127.0.0.1 esplicitamente invece di localhost
        self.send_socket = self.context.socket(zmq.REQ) # Modificato in REQ per stabilità standard, se fallisce rimettiamo CLIENT
        self.send_socket.connect("tcp://127.0.0.1:" + str(self.port))

       
        self.send_socket.close()
        try:
           
            self.send_socket = self.context.socket(zmq.CLIENT)
            self.send_socket.connect("tcp://127.0.0.1:" + str(self.port))

            self.rcv_socket = self.context.socket(zmq.SERVER)
            self.rcv_socket.bind("tcp://127.0.0.1:" + str(int(self.port)+1))
        except AttributeError:
            print("[!] La tua versione di pyzmq non supporta CLIENT/SERVER. Fallback su REQ/REP standard.")
            self.send_socket = self.context.socket(zmq.REQ)
            self.send_socket.connect("tcp://127.0.0.1:" + str(self.port))
            
            self.rcv_socket = self.context.socket(zmq.REP)
            self.rcv_socket.bind("tcp://127.0.0.1:" + str(int(self.port)+1))
        self.rcv_poller = zmq.Poller()
        self.rcv_poller.register(self.rcv_socket, zmq.POLLIN)

        self.bound = True
    
    def start(self):
        
        park_arg = os.path.join(BASE_PATH, "small_parks", "Electric Fields.SC6")
        
        args = [
            RCT_EXECUTABLE,
            park_arg, 
            "--port=" + self.port,
            # "--verbose" 
        ]
        
        if self.headless:
            args.append("--headless")

        print(f"[*] Avvio OpenRCT2 con comando: {' '.join(args)}")

    
        self.rct_process = subprocess.Popen(
            args, 
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            bufsize=1
        )

    
        print("[*] Avvio: Monitoraggio finestre di errore (60s)...")
        time.sleep(5)

        start_time = time.time()
        while time.time() - start_time < 60:
            found_error = False
            try:
                # Cerca finestre di errore zenity
                result = subprocess.run("xdotool search --class zenity", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    window_ids = result.stdout.strip().split('\n')
                    for wid in window_ids:
                        if wid:
                            print(f"[*] TROVATO ERRORE (ID: {wid}) -> INVIO DIRETTO")
                            # Invia Invio direttamente alla finestra
                            subprocess.run(f"xdotool key --window {wid} Return", shell=True)
                            found_error = True
            except Exception:
                pass
            
            if not found_error:
                time.sleep(1)
            else:
                time.sleep(2)
        
        print("[*] Monitoraggio terminato. Attesa caricamento...")
        # -------------------------------------------------------------------

    def get_output(self):
        
        pass

    def send_park(self, park_path):
        assert self.bound
        
        # Pulizia path
        if '"' in park_path: park_path = park_path.replace('"', '')
        if not os.path.isabs(park_path):
            park_path = os.path.join(BASE_PATH, park_path)

        print(f"[*] Richiesta caricamento parco: {park_path}")
        message = '{"action": "load_park", "path": "' + park_path + '"}'
        
        try:
            self.send_socket.send_string(message)
        except zmq.ZMQError as e:
            print(f"[!] Errore invio ZMQ: {e}")
            return "TIMEOUT"

        # FIX TIMEOUT: Aumentato drasticamente per il caricamento
        socks = dict(self.rcv_poller.poll(300000)) 
        
        if socks:
            if socks.get(self.rcv_socket) == zmq.POLLIN:
                pass
        else:
            print("[!] TIMEOUT in send_park (Il gioco non ha risposto in 60s)")
            
            return '{"status": "error", "msg": "timeout_loading_park"}'

        message = self.rcv_socket.recv()
        return message

    def send_action(self, action, **kwargs):
        assert self.bound
        
        d = {'action': action, **kwargs}
      
        message = str(d).replace("'", '"')
        
        self.send_socket.send_string(message)

        # Timeout variabili in base all'azione
        timeout_s = 60000 # 20s default
        if action == 'run_sim':
            timeout_s = 120000
        elif action == 'load_park':
            timeout_s = 300000

        socks = dict(self.rcv_poller.poll(timeout_s))
        if socks:
            if socks.get(self.rcv_socket) == zmq.POLLIN:
                pass
        else:
            print(f"[!] TIMEOUT in send_action ({action})")
            
            return '{"status": "error", "msg": "timeout_action"}'

        message = self.rcv_socket.recv()
        return message

    def capture_rct_window_to_file(self, filename):
        print(f"[Warn] capture_rct_window_to_file disabilitato su Linux/Headless ({filename})")
        pass
