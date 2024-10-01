from tools import enregistrer_fin_programme,lire_pid_du_fichier,git_commit_and_push, shutdown_system

enregistrer_fin_programme(lire_pid_du_fichier())
git_commit_and_push("2nd commit from python execution")
shutdown_system()
