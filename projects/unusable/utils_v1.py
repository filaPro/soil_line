import pandas


def list_channels(base_file_name):
    channels = {
        'blue': ['01', '02', '02'],
        'green': ['02', '03', '03'],
        'red': ['03', '04', '04'],
        'nir': ['04', '05', '08'],
        'swir1': ['05', '06', '11'],
        'swir2': ['07', '07', '12']
    }
    channel_shift = {
        'LT04': 0,
        'LT05': 0,
        'LE07': 0,
        'LC08': 1,
        'S2AB': 2
    }[base_file_name.split('_')[2]]
    return {channel: f'{base_file_name}_{channel}_{channels[channel][channel_shift]}.tif' for channel in channels}


def list_tif_files(path):
    return sorted(set('_'.join(file_name.split('_')[:4]) for file_name in os.listdir(path) if '.tif' in file_name))


def generate_or_read_labels(image_path, excel_path, fields):
    label_path = os.path.join(os.path.dirname(excel_path), 'labels.csv')
    # do nothing if labels.scv already exists
    if os.path.exists(label_path):
        return pandas.read_csv(label_path, index_col=0)

    base_file_names = list_tif_files(image_path)
    labels = pandas.DataFrame(0, index=base_file_names, columns=fields.names, dtype=np.uint8)
    # mark not intersecting fields



    # if training or validation
    if excel_path is not None:
        excel_file = pandas.read_excel(excel_path)
        excel_file['NDVI_map'] = excel_file['NDVI_map'].apply(lambda x: x[:-6])
        for item in excel_file.itertuples():
            if item[3] in labels.index and item[1] in labels.columns:
                labels.loc[item[3], item[1]] = 1
        labels.to_csv(label_path)