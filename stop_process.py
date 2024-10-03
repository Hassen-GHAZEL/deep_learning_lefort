import time
from tools import enregistrer_fin_programme,lire_pid_du_fichier,git_commit_and_push, shutdown_system

time.sleep(1*3600) # 1H00
enregistrer_fin_programme(lire_pid_du_fichier())
git_commit_and_push("commit from python execution, après 1H00 d'exécution")
shutdown_system()
