import os
from openpyxl import Workbook, load_workbook

class ExcelManager:
    def __init__(self, file_path: str, column_titles: list):
        """Initialise l'ExcelManager avec le chemin du fichier et les titres de colonne."""
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
        """Ajoute une nouvelle ligne à la feuille spécifiée. Crée la feuille si elle n'existe pas."""
        if sheet_name not in self.sheet_titles:
            # Crée une nouvelle feuille avec les titres de colonne
            sheet = self.workbook.create_sheet(title=sheet_name)
            sheet.append(self.column_titles)
            self.sheet_titles.append(sheet_name)
        else:
            # Accède à la feuille existante
            sheet = self.workbook[sheet_name]

        # Traitement des données de la ligne : remplacer '.' par ',' pour les floats
        processed_row = [
            str(item).replace('.', ',') if isinstance(item, tuple) else str(item).replace('.',',') if self.is_float_and_contains_dot(
                item) else item
            for item in row_data
        ]
        sheet.append(processed_row)  # Ajoute la ligne traitée
        self.save()  # Enregistre le fichier

    def get_last_row_first_column(self, sheet_name: str) -> int:
        """Renvoie la première colonne de la dernière ligne du sheet spécifié. Retourne 0 si le sheet n'existe pas."""
        if sheet_name not in self.sheet_titles:
            return 0  # Si la feuille n'existe pas, retourner 0

        sheet = self.workbook[sheet_name]  # Récupérer la feuille
        if sheet.max_row < 2:  # Vérifier si la feuille est vide ou ne contient que l'en-tête
            return 0

        last_row_value = sheet.cell(row=sheet.max_row, column=1).value  # Récupérer la valeur de la première colonne de la dernière ligne
        try:
            return int(last_row_value)  # Retourner la valeur convertie en int
        except (ValueError, TypeError):
            return 0  # Retourner 0 en cas d'échec de la conversion

    def count_rows(self, sheet_name: str) -> int:
        """Renvoie le nombre total de lignes dans la feuille spécifiée, y compris les en-têtes."""
        if sheet_name in self.sheet_titles:
            sheet = self.workbook[sheet_name]
            return sheet.max_row  # max_row retourne le nombre de lignes non vides
        return 0

    def read_rows(self, sheet_name: str):
        """Parcourt les lignes de la feuille spécifiée et renvoie les données sous forme de liste de listes."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Le fichier {self.file_path} n'existe pas.")  # Vérification de l'existence du fichier

        if sheet_name not in self.sheet_titles:
            for sheet in self.workbook.sheetnames:
                print(f"Feuille : {sheet}")  # Liste les feuilles existantes
            raise ValueError(f"La feuille {sheet_name} n'existe pas dans le fichier {self.file_path}.")

        sheet = self.workbook[sheet_name]  # Récupérer la feuille
        rows = [row for row in sheet.iter_rows(values_only=True)]  # Parcourir les lignes
        return rows  # Retourner les données lues

    @staticmethod
    def is_float_and_contains_dot(value):
        """Vérifie si une valeur est un float (ou peut être convertie en float) et contient un point décimal."""
        try:
            return isinstance(value, (float, int)) or (isinstance(value, str) and '.' in value and float(value))
        except ValueError:
            return False
