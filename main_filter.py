from Excel import ExcelManager
from time import time, sleep


def main():
    input_file = 'excel/deep_network_combinaison.xlsx'
    output_file = 'excel/deep_network_combinaison_filtered.xlsx'
    sheet_name = 'EVERYTHING'
    column_titles = ["numero epoch", "batch_size", "nb_epochs", "learning_rate", "input_size", "hidden_size", "output_size", "weight_init_range", "Training Loss", "Validation Loss", "Test Loss", "Accuracy"]

    manager = ExcelManager(input_file, column_titles)
    rows = manager.read_rows(sheet_name)

    # Ignorer la première ligne qui contient les titres des colonnes
    rows = rows[1:]

    filtered_manager = ExcelManager(output_file, column_titles)

    i_val_loss = column_titles.index("Validation Loss")
    i_train_loss = column_titles.index("Training Loss")
    i_num_epoch = column_titles.index("numero epoch")
    i_nb_epochs = column_titles.index("nb_epochs")


    i = 1
    while i < len(rows):
        current_row = rows[i]
        previous_row = rows[i - 1]
        epoque = 0
        print("ligne actuel : ", i + 1)
        if current_row[i_val_loss] > previous_row[i_val_loss] and current_row[i_train_loss] < previous_row[i_train_loss]:
            print(f"\t{current_row[i_num_epoch]} : {current_row[i_val_loss]} > {previous_row[i_val_loss]} et {current_row[i_train_loss]} < {previous_row[i_train_loss]}")
            filtered_manager.add_row(sheet_name, previous_row)
            epoque = previous_row[i_num_epoch]
            i += previous_row[i_nb_epochs] - (epoque)
        elif current_row[i_num_epoch] == current_row[i_nb_epochs]:
            print(f"\t{current_row[i_num_epoch]} : {current_row[i_num_epoch]} == {current_row[i_nb_epochs]}")
            filtered_manager.add_row(sheet_name, current_row)
            epoque = current_row[i_num_epoch]

        if epoque != 0:
            print(f"\tÉpoch de sauvegarde: {epoque}, pour le modele i={i}")

        i += 1


if __name__ == "__main__":
    main()