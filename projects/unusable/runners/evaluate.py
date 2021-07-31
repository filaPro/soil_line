import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, roc_auc_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--excel-path', type=str, default='/data/soil_line/unusable/fields_v2/flds_all_good.xls')
    parser.add_argument('--out-path', type=str, default='/data/soil_line/unusable/fields_v2/result.csv')
    options = parser.parse_args()

    predicted = pd.read_csv(options.out_path, index_col=0)
    excel_file = pd.read_excel(options.excel_path)
    # todo: .xls -> .csv and remove apply
    excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:-4])
    true = pd.DataFrame(0, index=predicted.index, columns=predicted.columns)
    for _, row in excel_file.iterrows():
        if row['NDVI_map'] in true.index and row['name'] in true.columns:
            true.loc[row['NDVI_map'], row['name']] = 1

    print('accuracy', accuracy_score(true.values.reshape(-1), predicted.values.reshape(-1) >= .5))
    print('auc', roc_auc_score(true.values.reshape(-1), predicted.values.reshape(-1)))
