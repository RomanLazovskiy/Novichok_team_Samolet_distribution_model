
def device(data):
    columns = [f'col{n}' for n in range(529, 544)]
    new_feature_name = 'device'
    most_common_value = data[columns].mode(axis=1).iloc[:, 0]
    data[new_feature_name] = most_common_value
    return data

def info_redirect(data):
    columns = [f'col{n}' for n in range(561, 576)]
    mail_info = 'info_mail'
    most_common_value = any(['sms' in data[columns], 'email' in data[columns]])
    data[mail_info] = int(most_common_value)
    
    invite_count = 'sale_invite_count'
    most_common_value = (data[columns].notnull()).sum(axis=1)
    data[invite_count] = most_common_value
    return data

def generate_new_features(data):
    """
    Генерирует новые признаки на основе заданных колонок.
    :param data: pandas.DataFrame, содержащий данные.
    :param columns: список, содержащий названия колонок для генерации новых признаков.
    :return: pandas.DataFrame с добавленными новыми признаками.
    """
    columns = [f'col{n}' for n in range(465, 473)]
    new_feature_name = 'filter' + '_most_common'
    most_common_value = data[columns].mode(axis=1).iloc[:, 0]
    data[new_feature_name] = most_common_value

    count_feature_name = 'filter' + '_count'
    count_of_most_common = (data[columns] == most_common_value[..., None]).sum(axis=1)
    data[count_feature_name] = count_of_most_common

    non_zero_feature_name = 'filter' + '_non_zero_count'
    non_zero_count = (data[columns] != 0).sum(axis=1)
    data[non_zero_feature_name] = non_zero_count

    return data

def calculate_quantitative_metrics(data):
    """
    Вычисляет количественные метрики на основе заданных колонок для каждой строки в датафрейме.
    :param data: pandas.DataFrame, содержащий данные.
    :param columns: список, содержащий названия колонок для вычисления метрик.
    :return: pandas.DataFrame с добавленными количественными метриками.
    """
    columns = [f'col{n}' for n in range(457,465)]
    # Количество уникальных страниц
    unique_value = data[columns].nunique(axis=1)
    data['unique_pages_count'] = unique_value

    # Количество общих страниц
    count_feature_name = 'visit_site' + '_count'
    count_of_most_common = (data[columns].notnull()).sum(axis=1)
    data[count_feature_name] = count_of_most_common

    return data