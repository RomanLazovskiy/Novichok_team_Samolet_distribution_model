import pandas as pd
from build_features import device, info_redirect, generate_new_features, calculate_quantitative_metrics

def inference_model(df, models, prob_col='score'):
    preds = []
    for m in models:
        preds.append(m.predict_proba(df_test)[:, 1])
    df[prob_col] = pd.DataFrame(preds).mean()
    return df


def get_submission(path_to_csv, name_save_file='result_novichki.csv'):
    test = pd.read_csv(path_to_csv)
    
    for func_feature in [device, info_redirect, generate_new_features, calculate_quantitative_metrics]:
        test = func_feature(test)
        
    test = inference_model(test, models)
    sample_submission = test.loc[:, ['report_date', 'client_id', 'score']]
    sample_submission.to_csv(name_save_file)