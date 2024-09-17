import os
from openpyxl import Workbook, load_workbook

class ExcelManager:
    def __init__(self, file_path: str, column_titles: list):
        self.file_path = file_path
        self.column_titles = column_titles
        self.sheet_titles = []

        # Vérification de l'existence du fichier
        if os.path.exists(file_path):
            self.workbook = load_workbook(file_path)
            self.sheet_titles = self.workbook.sheetnames
        else:
            # Création du fichier et de la première feuille par défaut
            self.workbook = Workbook()
            self.workbook.active.title = "Sheet1"  # Renommer la feuille par défaut
            self.save()

    def save(self):
        """Enregistre le fichier Excel."""
        self.workbook.save(self.file_path)

    def add_row(self, sheet_name: str, row_data: list):
        """Ajoute une nouvelle ligne à la feuille spécifiée. Si la feuille n'existe pas, elle est créée avec les titres de colonne."""
        if sheet_name not in self.sheet_titles:
            print(f"typeof(self.column_titles) : {type(self.column_titles)}")
            sheet = self.workbook.create_sheet(title=sheet_name)
            sheet.append(self.column_titles)
            self.sheet_titles.append(sheet_name)
        else:
            sheet = self.workbook[sheet_name]

        # Traitement des données de la ligne : remplacer '.' par ',' uniquement pour les floats avec un point
        processed_row = [
            str(item).replace('.', ',') if isinstance(item, tuple) else str(item).replace('.',
                                                                                          ',') if self.is_float_and_contains_dot(
                item) else item
            for item in row_data
        ]

        print(f"type(processed_row) {type(processed_row)}, processed_row : {processed_row}")
        sheet.append(processed_row)
        self.save()

    @staticmethod
    def is_float_and_contains_dot(value):
        """Vérifie si une valeur est un float (ou peut être convertie en float) et contient un point décimal."""
        try:
            # On essaie d'analyser la valeur pour voir si elle est un float
            return isinstance(value, (float, int)) or (isinstance(value, str) and '.' in value and float(value))
        except ValueError:
            return False
