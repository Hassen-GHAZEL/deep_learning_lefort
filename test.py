
from tools import shutdown_system
import time

for i in range(60):
    print(f"Le programme s'exécute depuis {i} secondes.")
    time.sleep(1)
shutdown_system()

