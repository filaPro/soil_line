import os
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, roc_auc_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in-path', type=str, default='/volume/soil_line/unusable')
    parser.add_argument('--out-path', type=str, default='/volume/logs/unusable/...')
    options = vars(parser.parse_args())

    predicted = pd.read_csv(os.path.join(options['out_path'], 'result.csv'), index_col=0)
    excel_file = pd.read_excel(os.path.join(options['in_path'], 'NDVI_list.xls'))
    excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:-6])
    true = pd.DataFrame(0, index=predicted.index, columns=predicted.columns)
    for item in excel_file.itertuples():
        if item[3] in true.index and item[1] in true.columns:
            true.loc[item[3], item[1]] = 1

    print('accuracy', accuracy_score(true.values.reshape(-1), predicted.values.reshape(-1) >= .5))
    print('auc', roc_auc_score(true.values.reshape(-1), predicted.values.reshape(-1)))
