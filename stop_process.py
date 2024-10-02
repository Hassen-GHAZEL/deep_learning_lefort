import time
from tools import enregistrer_fin_programme,lire_pid_du_fichier,git_commit_and_push, shutdown_system

time.sleep(3600 * 5)
enregistrer_fin_programme(lire_pid_du_fichier())
git_commit_and_push("commit from python execution, après 5 heures d'exécution")
shutdown_system()
