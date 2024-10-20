from tools import save_boxplot_with_stats

def main():
    # Chemin du fichier Excel
    excel_file = "excel/shallow_network_combinaison.xlsx"
    sheet_name = "EVERYTHING"  # Remplacez par le nom de votre feuille si nécessaire
    column_name = "Accuracy"
    output_file = "image/boxplot.png"

    try:
        save_boxplot_with_stats(excel_file, sheet_name, column_name, output_file)
        print(f"Boxplot sauvegardé dans {output_file}.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()