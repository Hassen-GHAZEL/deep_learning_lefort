import time
from datetime import datetime
from tools import get_gpu_temperature, shutdown_system, create_or_overwrite_file

temperature = 0
while True:
    temperature = get_gpu_temperature()
    print(f"La température actuelle du GPU est de {temperature}°C.")
    if temperature >= 65:
        break # Arrêter la boucle si la température dépasse 65°C
    time.sleep(60)  # Attendre 1 minute avant de vérifier à nouveau la température

msg = f"program a commencer a 15:40:44 et a fini a {datetime.now().strftime('%H:%M:%S')} et la temperature du GPU est de {temperature}°C"
create_or_overwrite_file("duree_totale.txt", msg)
shutdown_system()
"""

tab_batch_size = [2, 4, 8, 16, 32, 64, 128]
tab_learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75, 1, 2]
tab_hidden_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
tab_weight_init_range = [(-0.0001, 0.0001), (-0.001, 0.001), (-0.01, 0.01), (-0.1, 0.1), (-1, 1), (-2, 2), (-5, 5), (-10, 10)]

print(f"{len(tab_batch_size)} + {len(tab_learning_rate)} + {len(tab_hidden_size)} + {len(tab_weight_init_range)} = {len(tab_batch_size) + len(tab_learning_rate) + len(tab_hidden_size) + len(tab_weight_init_range)}")

"""