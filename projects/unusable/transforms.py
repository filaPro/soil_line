def get_day(base_file_name):
    string = base_file_name.split('_')[1]
    day = int(string[:2]) * 30 + int(string[2:])
    return day


def get_satellite(base_file_name):
    if 'LT04' in base_file_name:
        return 4
    elif 'LT05' in base_file_name:
        return 5
    elif 'LE07' in base_file_name:
        return 7
    elif 'LC08' in base_file_name:
        return 8
    raise ValueError(f'Unknown satellite for file: {base_file_name}')


def catboost_transform(images, base_file_name, mask, field_name, x, y, label):
    return {
        'day': get_day(base_file_name),
        'satellite': get_satellite(base_file_name),
        'label': label
    }

