import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt


def field_averaged_metric(true_, pred_, quantile=0.9):
    fields = true_.keys()

    res_true = []
    res_pred = []
    for field in fields:
        ind = np.argsort(-pred_[field].values)
        res_true.append(np.take(true_[field].values, ind))
        res_pred.append(np.take(pred_[field].values, ind))

    res_true = np.array(res_true)

    number_to_view = np.nanquantile(res_true *
                                    [range(res_true.shape[1])] /
                                    (res_true > 0), quantile, axis=1)
    number_positive = res_true.sum(axis=1)
    return {'number_to_view': number_to_view.sum(),
            'number_positive': number_positive.sum()}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--excel-path', type=str, default='/data/soil_line/unusable/fields_v2/flds_all_good.xls')
    parser.add_argument('--out-path', type=str, default='/data/soil_line/unusable/fields_v2/result.csv')
    options = parser.parse_args()

    predicted = pd.read_csv(options.out_path, index_col=0)
    excel_file = pd.read_excel(options.excel_path)
    # todo: .xls -> .csv and remove apply
    excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:22])  # yDDDD_DDDD_LTDD_DDDDDD
    true = pd.DataFrame(0, index=predicted.index, columns=predicted.columns)
    for _, row in excel_file.iterrows():
        if row['NDVI_map'] in true.index and row['name'] in true.columns:
            true.loc[row['NDVI_map'], row['name']] = 1

    print(field_averaged_metric(true, predicted))

    pred = predicted.values.reshape(-1)
    true = true.values.reshape(-1)

    pred05 = pred > .5

    fp = sum(pred05 * (1 - true))
    fn = sum((~pred05) * true)
    tp = sum(pred05 * true)
    tn = sum((~pred05) * (1 - true))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    roc_auc = roc_auc_score(true, pred)

    stats = {
        'recall': recall,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

    print(json.dumps(stats, indent=4))
    with open(os.path.join(os.path.dirname(options.out_path), 'stats.json'), 'w') as file:
        json.dump(stats, file, indent=4)

    PrecisionRecallDisplay.from_predictions(true, pred)
    plt.savefig(os.path.join(os.path.dirname(options.out_path), 'precision_recall_curve.png'))

    RocCurveDisplay.from_predictions(true, pred)
    plt.savefig(os.path.join(os.path.dirname(options.out_path), 'roc_curve.png'))
