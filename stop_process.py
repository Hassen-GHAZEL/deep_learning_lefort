import time
from tools import enregistrer_fin_programme,lire_pid_du_fichier,git_commit_and_push, shutdown_system

time.sleep(5*3600) # 5H05
enregistrer_fin_programme(lire_pid_du_fichier())
git_commit_and_push("commit from python execution, après 05h00 d'exécution")
shutdown_system()
