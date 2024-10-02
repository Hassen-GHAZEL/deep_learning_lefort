from tools import *


temperature = 0
while True:
    temperature = get_gpu_temperature()
    print(f"La température actuelle du GPU est de {temperature}°C.")
    if temperature >= 65:
        break # Arrêter la boucle si la température dépasse 65°C
    time.sleep(60)  # Attendre 1 minute avant de vérifier à nouveau la température

msg = f"program a fini a {datetime.now().strftime('%H:%M:%S')} et la temperature du GPU est de {temperature}°C"
create_or_overwrite_file("txt/info_surchauffe.txt", msg)
enregistrer_fin_programme(pid = lire_pid_du_fichier())
git_commit_and_push(f"commit temperature de securite: {temperature}°C")
shutdown_system()

