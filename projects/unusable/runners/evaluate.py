import os
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, accuracy_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--excel-path', type=str, default='/data/soil_line/unusable/fields_v2/flds_all_good.xls')
    parser.add_argument('--out-path', type=str, default='/data/soil_line/unusable/.../result.csv')
    options = parser.parse_args()

    predicted = pd.read_csv(options.out_path, index_col=0)
    excel_file = pd.read_excel(options.excel_path)
    # todo: .xls -> .csv and remove apply
    excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:22])  # yDDDD_DDDD_LTDD_DDDDDD
    true = pd.DataFrame(0, index=predicted.index, columns=predicted.columns)
    for _, row in excel_file.iterrows():
        if row['NDVI_map'] in true.index and row['name'] in true.columns:
            true.loc[row['NDVI_map'], row['name']] = 1

    pred = predicted.values.reshape(-1)
    true = true.values.reshape(-1)

    print('accuracy:', accuracy_score(true, pred > .5))
    print('auc:', roc_auc_score(true, pred))

    curve = PrecisionRecallDisplay.from_predictions(true, pred)
    for i in range(len(curve.recall)):
        if curve.recall[i] < curve.precision[i]:
            print('curve equals at recall:', curve.recall[i], 'and precision:', curve.precision[i])
            break
    plt.savefig(os.path.join(os.path.dirname(options.out_path), 'precision_recall_curve.png'))

    RocCurveDisplay.from_predictions(true, pred)
    plt.savefig(os.path.join(os.path.dirname(options.out_path), 'roc_curve.png'))
