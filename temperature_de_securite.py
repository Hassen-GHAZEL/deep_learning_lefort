from tools import *


temperature = 0
while True:
    temperature = get_gpu_temperature()
    print(f"La température actuelle du GPU est de {temperature}°C.")
    if temperature >= 60:
        break # Arrêter la boucle si la température dépasse 65°C
    time.sleep(60)  # Attendre 1 minute avant de vérifier à nouveau la température

msg = f"commit à {datetime.now().strftime('%H:%M:%S')} et GPU temperature = {temperature}°C"
enregistrer_fin_programme(pid = lire_pid_du_fichier())
git_commit_and_push(msg)
shutdown_system()

