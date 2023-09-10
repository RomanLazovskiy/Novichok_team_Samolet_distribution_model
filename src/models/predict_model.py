import pandas as pd
from src.features.build_features import device, info_redirect, generate_new_features, calculate_quantitative_metrics

def inference_model(input_dataset, models, prob_col='score'):
    for func_feature in [device, info_redirect, generate_new_features, calculate_quantitative_metrics]:
        input_dataset = func_feature(input_dataset)
    preds = []
    for m in models:
        preds.append(m.predict_proba(input_dataset)[:, 1])
    input_dataset[prob_col] = pd.DataFrame(preds).mean()
    result_df = input_dataset[['client_id', 'report_date', prob_col]]
    return result_df


def get_submission(path_to_csv, models, name_save_file='result_novichki.csv'):
    test = pd.read_csv(path_to_csv)
    
    for func_feature in [device, info_redirect, generate_new_features, calculate_quantitative_metrics]:
        test = func_feature(test)
        
    test = inference_model(test, models)
    sample_submission = test.loc[:, ['report_date', 'client_id', 'score']]
    sample_submission.to_csv(name_save_file)