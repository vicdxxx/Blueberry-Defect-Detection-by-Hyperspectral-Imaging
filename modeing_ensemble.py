from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
import time
from modeling_spectral_data import load_spectral_mean_data_xy
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
"""ensemble_method:
    statistics_driven
    fuzzy_logic
"""
# shrinkage_spectral=6e-6, shrinkage_spatial=0.363
# shrinkage_spectral=5e-6, shrinkage_spatial=1e-6

def show_statistics(x, reshape=False):
    xxx = np.array(x)
    if reshape:
        xxx = xxx.reshape(-1)
    df_describe = pd.DataFrame(xxx)
    print(df_describe.describe())

def repeated_test_ensamble(model_spectral, model_spatial, feature1, feature2, label, test_num=100, seed=None, model_params_spectral=None, model_params_spatial=None, ensemble_method='statistics_driven'):
    from tqdm import tqdm
    accuracy_spectrals = []
    accuracy_spatials = []
    accuracy_ensemble = []

    precision_spectrals = []
    precision_spatials = []
    precision_ensemble = []

    recall_spectrals = []
    recall_spatials = []
    recall_ensemble = []

    all_score_spectrals_train = []
    all_score_spatials_train = []

    all_score_spectral_norms_train = []
    all_score_spatial_norms_train = []

    all_pred_spectral_norms_train = []
    all_pred_spatial_norms_train = []

    all_score_spectrals_test = []
    all_score_spatials_test = []

    all_score_spectral_norms_test = []
    all_score_spatial_norms_test = []

    all_pred_spectral_norms_test = []
    all_pred_spatial_norms_test = []

    all_y_test = []
    all_y_train = []

    all_bias = []

    if 'PLS' in model_spectral.__name__ or 'PLS' in model_spatial.__name__:
        boundary_center = 0.5

    # print('feature1.shape', feature1.shape)
    # print('feature2.shape', feature2.shape)
    for i_test in tqdm(range(test_num)):
        if seed is None:
            np.random.seed(int(time.time()))
        random_state = np.random.randint(0, 1e4)
        X_train, X_test, y_train, y_test = train_test_split(feature1, label, test_size=0.30, random_state=random_state, shuffle=True, stratify=label)
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        if type(feature2) is list:
            X_train1_list = []
            X_test1_list = []
            y_train1_list = []
            y_test1_list = []
            for feature2_ in feature2:
                X_train1_, X_test1_, y_train1_, y_test1_ = train_test_split(
                    feature2_, label, test_size=0.30, random_state=random_state, shuffle=True, stratify=label)
                y_train1_ = y_train1_.squeeze()
                y_test1_ = y_test1_.squeeze()
                X_train1_list.append(X_train1_)
                X_test1_list.append(X_test1_)
                y_train1_list.append(y_train1_)
                y_test1_list.append(y_test1_)
        else:
            X_train1, X_test1, y_train1, y_test1 = train_test_split(feature2, label, test_size=0.30, random_state=random_state, shuffle=True, stratify=label)
            y_train1 = y_train1.squeeze()
            y_test1 = y_test1.squeeze()

        if model_params_spectral is not None:
            model_ = model_spectral(**model_params_spectral)
        else:
            model_ = model_spectral()
        if 'PLS' in model_spectral.__name__:
            y_train = pd.get_dummies(y_train)

        model_.fit(X_train, y_train)
        predictions = model_.predict(X_test)

        if 'PLS' in model_spectral.__name__:
            predictions = np.array([np.argmax(i) for i in predictions])

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        diffs = np.where(predictions != y_test.values.squeeze())
        accuracy_spectral = 1-len(diffs[0])/len(y_test)

        precision_spectrals.append(precision)
        recall_spectrals.append(recall)
        accuracy_spectrals.append(accuracy_spectral)
        if 'PLS' in model_spectral.__name__:
            score_spectral_train = model_.predict(X_train)
            score_spectral_train = np.max(score_spectral_train, 1) - boundary_center
            score_spectral_train[score_spectral_train <= 0] = 0
        else:
            score_spectral_train = model_.decision_function(X_train)
        df_describe = pd.DataFrame(score_spectral_train)
        statistics = df_describe.describe()
        #mean_spectral_train = statistics.loc['mean'].values
        #std_spectral_train = statistics.loc['std'].values
        mean_spectral_train = np.array(score_spectral_train).reshape(-1).mean()
        std_spectral_train = np.array(score_spectral_train).reshape(-1).std()

        # model_params1 ={'shrinkage': 0.35, 'solver':'eigen'}
        # model_params1 ={'shrinkage': 0.62, 'solver':'eigen'}
        # model_params1 ={'shrinkage': 0.363, 'solver':'eigen'}
        # model_params1 ={'shrinkage': 0.5349, 'solver':'eigen'}
        if type(feature2) is list:
            score_spatial_train = None
            precision_spatial_bands = []
            recall_spatial_bands = []
            accuracy_spatial_bands = []
            model_spatial_all = []
            assert type(model_params_spectral) is list
            for i_model, model_param in enumerate(model_params_spectral):
                model_params1 = model_param
                if model_params1 is not None:
                    model_1 = model_spatial(**model_params1)
                else:
                    model_1 = model_spatial()
                if 'PLS' in model_spatial.__name__:
                    y_train1_list[i_model] = pd.get_dummies(y_train1_list[i_model])

                model_1.fit(X_train1_list[i_model], y_train1_list[i_model])
                predictions1 = model_1.predict(X_test1_list[i_model])

                if 'PLS' in model_spatial.__name__:
                    predictions1 = np.array([np.argmax(i) for i in predictions1])

                precision = precision_score(y_test1_list[i_model], predictions1)
                recall = recall_score(y_test1_list[i_model], predictions1)
                diffs = np.where(predictions1 != y_test1_list[i_model].values.squeeze())
                accuracy_spatial = 1-len(diffs[0])/len(y_test1_list[i_model])
                precision_spatial_bands.append(precision)
                recall_spatial_bands.append(recall)
                accuracy_spatial_bands.append(accuracy_spatial)

                if score_spatial_train is None:
                    score_spatial_train = model_1.decision_function(X_train1_list[i_model])
                else:
                    score_spatial_train += model_1.decision_function(X_train1_list[i_model])
                model_spatial_all.append(model_1)

            precision_spatials.append(precision_spatial_bands)
            recall_spatials.append(recall_spatial_bands)
            accuracy_spatials.append(accuracy_spatial_bands)

            df_describe = pd.DataFrame(score_spatial_train)
            statistics = df_describe.describe()
            mean_spatial_train = statistics.loc['mean'].values
            std_spatial_train = statistics.loc['std'].values
        else:
            model_params1 = model_params_spatial
            if model_params1 is not None:
                model_1 = model_spatial(**model_params1)
            else:
                model_1 = model_spatial()
            if 'PLS' in model_spatial.__name__:
                y_train1 = pd.get_dummies(y_train1)

            model_1.fit(X_train1, y_train1)
            predictions1 = model_1.predict(X_test1)

            if 'PLS' in model_spatial.__name__:
                predictions1 = np.array([np.argmax(i) for i in predictions1])

            precision = precision_score(y_test1, predictions1)
            recall = recall_score(y_test1, predictions1)
            diffs = np.where(predictions1 != y_test1.values.squeeze())
            accuracy_spatial = 1-len(diffs[0])/len(y_test1)
            if 'PLS' in model_spatial.__name__:
                score_spatial_train = model_1.predict(X_train1)
                score_spatial_train = np.max(score_spatial_train, 1) - boundary_center
                score_spatial_train[score_spatial_train <= 0] = 0
            else:
                score_spatial_train = model_1.decision_function(X_train1)

            df_describe = pd.DataFrame(score_spatial_train)
            statistics = df_describe.describe()
            #mean_spatial_train = statistics.loc['mean'].values
            #std_spatial_train = statistics.loc['std'].values
            mean_spatial_train = np.array(score_spatial_train).reshape(-1).mean()
            std_spatial_train = np.array(score_spatial_train).reshape(-1).std()

            precision_spatials.append(precision)
            recall_spatials.append(recall)
            accuracy_spatials.append(accuracy_spatial)

        score_spectrals = []
        score_spatials = []
        score_spectral_norms = []
        score_spatial_norms = []
        pred_spectral_norms = []
        pred_spatial_norms = []
        for i_score, _ in enumerate(score_spectral_train):
            score_spectral_norm = (score_spectral_train[i_score]-mean_spectral_train)/std_spectral_train
            score_spatial_norm = (score_spatial_train[i_score]-mean_spatial_train)/std_spatial_train

            score_spectrals.append(score_spectral_train[i_score])
            score_spatials.append(score_spatial_train[i_score])

            score_spectral_norms.append(score_spectral_norm)
            score_spatial_norms.append(score_spatial_norm)

            pred_spectral_norm = 1 if score_spectral_norm > 0 else 0
            pred_spatial_norm = 1 if score_spatial_norm > 0 else 0

            pred_spectral_norms.append(pred_spectral_norm)
            pred_spatial_norms.append(pred_spatial_norm)

        score_spectrals = np.array(score_spectrals)
        score_spatials = np.array(score_spatials)
        all_score_spectrals_train.append(score_spectrals)
        all_score_spatials_train.append(score_spatials)

        score_spectral_norms = np.array(score_spectral_norms)
        score_spatial_norms = np.array(score_spatial_norms)
        all_score_spectral_norms_train.append(score_spectral_norms)
        all_score_spatial_norms_train.append(score_spatial_norms)

        pred_spectral_norms = np.array(pred_spectral_norms)
        pred_spatial_norms = np.array(pred_spatial_norms)
        all_pred_spectral_norms_train.append(pred_spectral_norms)
        all_pred_spatial_norms_train.append(pred_spatial_norms)

        all_y_train.append(y_train)
        if 'PLS' in model_spectral.__name__:
            score_spectral_test = model_.predict(X_test)
            score_spectral_test = np.max(score_spectral_test, 1) - boundary_center
            score_spectral_test[score_spectral_test <= 0] = 0
        else:
            score_spectral_test = model_.decision_function(X_test)

        if type(feature2) is list:
            score_spatial_test = None
            for i_model, _ in enumerate(X_test1_list):
                if score_spatial_test is None:
                    score_spatial_test = model_spatial_all[i_model].decision_function(X_test1_list[i_model])
                else:
                    score_spatial_test += model_spatial_all[i_model].decision_function(X_test1_list[i_model])
        else:
            if 'PLS' in model_spatial.__name__:
                score_spatial_test = model_1.predict(X_test1)
                score_spatial_test = np.max(score_spatial_test, 1) - boundary_center
                score_spatial_test[score_spatial_test <= 0] = 0
            else:
                score_spatial_test = model_1.decision_function(X_test1)

        score_spectrals = []
        score_spatials = []
        score_spectral_norms = []
        score_spatial_norms = []
        pred_spectral_norms = []
        pred_spatial_norms = []

        predictions_final = []
        if type(feature2) is list:
            bias = np.array(score_spatial_train).reshape(-1).std() / len(feature2) / np.array(score_spectral_train).reshape(-1).std()
        else:
            #bias = np.array(score_spatial_train).reshape(-1).std() / np.array(score_spectral_train).reshape(-1).std()
            bias = std_spatial_train / std_spectral_train
        #bias = mean_spatial_train - mean_spectral_train
        all_bias.append(bias)
        for i_score, _ in enumerate(score_spectral_test):
            score_spectral_norm = (score_spectral_test[i_score]-mean_spectral_train)/std_spectral_train
            score_spatial_norm = (score_spatial_test[i_score]-mean_spatial_train)/std_spatial_train

            #score_spectral_norm = (score_spectral_test[i_score]-mean_spectral_train)
            #score_spatial_norm = (score_spatial_test[i_score]-mean_spatial_train)

            score_spectrals.append(score_spectral_test[i_score])
            score_spatials.append(score_spatial_test[i_score])

            score_spectral_norms.append(score_spectral_norm)
            score_spatial_norms.append(score_spatial_norm)

            pred_spectral_norm = 1 if score_spectral_norm > 0 else 0
            pred_spatial_norm = 1 if score_spatial_norm > 0 else 0

            pred_spectral_norms.append(pred_spectral_norm)
            pred_spatial_norms.append(pred_spatial_norm)

            if np.sign(score_spectral_norm) != np.sign(score_spatial_norm):
                if ensemble_method == 'statistics_driven':
                    # this bias more like SVM (hinge loss) than fuzzy logic
                    # bias = 0.22

                    #if abs(score_spectral_norm) > abs(score_spatial_norm)*bias:
                    #if abs(score_spectral_norm)+bias > abs(score_spatial_norm):
                    #    final_score = score_spectral_norm
                    #else:
                    #    final_score = score_spatial_norm

                    if score_spatial_norm > 0:
                        if abs(score_spectral_norm) + 1.1*bias > abs(score_spatial_norm):
                            final_score = score_spectral_norm
                        else:
                            final_score = score_spatial_norm
                    else:
                        final_score = score_spectral_norm

                # if ensemble_method == 'svc_model':
                    # res = svc_model.predict([[score_spectral_norm[0], score_spatial_norm[0]]])
                    # final_score = score_spectral_norm if res ==0 else score_spatial_norm
                elif ensemble_method == 'fuzzy_logic':
                    # bias=0.62
                    spectral_value = ((score_spectral_norm/3.0)+1.0)/2.0
                    spatial_value = ((score_spatial_norm/3.0)+1.0)/2.0
                    final_score = fuzzy_logic_infer(spectral_value, spatial_value, bias=3*bias)
                else:
                    assert False
            else:
                final_score = score_spectral_norm

            if final_score > 0:
                final_cls = 1
            else:
                final_cls = 0
            predictions_final.append(final_cls)

        score_spectrals = np.array(score_spectrals)
        score_spatials = np.array(score_spatials)
        all_score_spectrals_test.append(score_spectrals)
        all_score_spatials_test.append(score_spatials)

        score_spectral_norms = np.array(score_spectral_norms)
        score_spatial_norms = np.array(score_spatial_norms)
        all_score_spectral_norms_test.append(score_spectral_norms)
        all_score_spatial_norms_test.append(score_spatial_norms)

        pred_spectral_norms = np.array(pred_spectral_norms)
        pred_spatial_norms = np.array(pred_spatial_norms)
        all_pred_spectral_norms_test.append(pred_spectral_norms)
        all_pred_spatial_norms_test.append(pred_spatial_norms)

        all_y_test.append(y_test)

        precision = precision_score(y_test, predictions_final)
        recall = recall_score(y_test, predictions_final)

        diffs = np.where(predictions_final != y_test.values.squeeze())
        final_accuracy = 1-len(diffs[0])/len(y_test)

        precision_ensemble.append(precision)
        recall_ensemble.append(recall)
        accuracy_ensemble.append(final_accuracy)
    return precision_spectrals, precision_spatials, precision_ensemble, recall_spectrals, recall_spatials, recall_ensemble, accuracy_spectrals, accuracy_spatials, accuracy_ensemble, all_y_train, all_score_spectral_norms_train, all_score_spatial_norms_train, all_pred_spectral_norms_train, all_pred_spatial_norms_train, all_score_spectrals_train, all_score_spatials_train, all_y_test, all_score_spectral_norms_test, all_score_spatial_norms_test, all_pred_spectral_norms_test, all_pred_spatial_norms_test, all_score_spectrals_test, all_score_spatials_test, all_bias


def main():
    band_list = [413.52, 599.97, 712.06, 736.18, 812.8, 911.41]
    spatial_features_num_dict = {
        'haralick_features': 168, 'lbp_features': 354, 'hu_moments_features': 42, 'gabor_features': 402, 'bsif_features': 1536,
        'order': ['haralick_features', 'lbp_features', 'hu_moments_features', 'gabor_features', 'bsif_features']
    }

    from modeling_spectral_data import load_im_dirs
    im_dirs = load_im_dirs()
    len(im_dirs)

    save_path_all_spatial_features = r'D:\BoyangDeng\BlueberryClassification\datasets\mean_data_python\all_spatial_features_6bands.npy'

    all_spatial_features = np.load(save_path_all_spatial_features, allow_pickle='TRUE')

    spatial_feature_means = all_spatial_features.mean(axis=0)
    spatial_feature_stds = all_spatial_features.std(axis=0)

    all_spatial_features_norm = (all_spatial_features - spatial_feature_means) / spatial_feature_stds
    all_spatial_features.mean(0)[:10], all_spatial_features.std(0)[:10], all_spatial_features_norm.mean(0)[:10], all_spatial_features_norm.std(0)[:10]

    x, y = load_spectral_mean_data_xy(use_part_1=1, use_part_2=1)
    X = pd.DataFrame(x)
    label = pd.DataFrame(y, columns=['label'])

    X = x
    X_norm = (x-x.mean(0))/x.std(0)

    """
    average
    """
    im_num_each_sample = len(band_list)
    haralick_num = spatial_features_num_dict['haralick_features']
    lbp_num = spatial_features_num_dict['lbp_features']
    hu_num = spatial_features_num_dict['hu_moments_features']
    gabor_num = spatial_features_num_dict['gabor_features']
    bsif_num = spatial_features_num_dict['bsif_features']
    # haralick_num + lbp_num + hu_num + gabor_num + bsif_num
    start = 0

    spatial_features_order = spatial_features_num_dict['order']

    haralick_features = all_spatial_features[:, start: start+haralick_num]
    haralick_features_mean = haralick_features.reshape(-1, haralick_num//im_num_each_sample, im_num_each_sample).mean(2)

    lbp_features = all_spatial_features[:, start+haralick_num: start+haralick_num+lbp_num]
    lbp_features_mean = lbp_features.reshape(-1, lbp_num//im_num_each_sample, im_num_each_sample).mean(2)

    hu_moments_features = all_spatial_features[:, start+haralick_num+lbp_num: start+haralick_num+lbp_num+hu_num]
    hu_moments_features_mean = hu_moments_features.reshape(-1, hu_num//im_num_each_sample, im_num_each_sample).mean(2)

    gabor_features = all_spatial_features[:, start+haralick_num+lbp_num+hu_num: start+haralick_num+lbp_num+hu_num+gabor_num]
    gabor_features_mean = gabor_features.reshape(-1, gabor_num//im_num_each_sample, im_num_each_sample).mean(2)

    bsif_features = all_spatial_features[:, start+haralick_num+lbp_num+hu_num+gabor_num: start+haralick_num+lbp_num+hu_num+gabor_num+bsif_num]
    bsif_features_mean = bsif_features.reshape(-1, bsif_num//im_num_each_sample, im_num_each_sample).mean(2)

    haralick_features_norm = (haralick_features - haralick_features.mean(axis=0))/haralick_features.std(axis=0)
    lbp_features_norm = (lbp_features - lbp_features.mean(axis=0))/lbp_features.std(axis=0)
    hu_moments_features_norm = (hu_moments_features - hu_moments_features.mean(axis=0))/hu_moments_features.std(axis=0)
    gabor_features_norm = (gabor_features - gabor_features.mean(axis=0))/gabor_features.std(axis=0)
    bsif_features_norm = (bsif_features - bsif_features.mean(axis=0))/bsif_features.std(axis=0)

    haralick_features_norm_mean = (haralick_features_mean - haralick_features_mean.mean(axis=0))/haralick_features_mean.std(axis=0)
    lbp_features_norm_mean = (lbp_features_mean - lbp_features_mean.mean(axis=0))/lbp_features_mean.std(axis=0)
    hu_moments_features_norm_mean = (hu_moments_features_mean - hu_moments_features_mean.mean(axis=0))/hu_moments_features_mean.std(axis=0)
    gabor_features_norm_mean = (gabor_features_mean - gabor_features_mean.mean(axis=0))/gabor_features_mean.std(axis=0)
    bsif_features_norm_mean = (bsif_features_mean - bsif_features_mean.mean(axis=0))/bsif_features_mean.std(axis=0)

    # best combination? [haralick_features, lbp_features, hu_moments_features]
    spatial_features = np.concatenate([haralick_features, lbp_features, hu_moments_features, gabor_features, bsif_features], axis=1)
    spatial_features_norm = np.concatenate([haralick_features_norm, lbp_features_norm, hu_moments_features_norm,
                                           gabor_features_norm, bsif_features_norm], axis=1)

    shrinkage_spectral = 5e-6
    shrinkage_spatial = 'auto'
    # i_select=8
    # feature2 = selected_features_list[i_select]
    # combine = combine_list_sort[i_select]
    # shrinkage_spatial = good_params_list_sort[i_select]['shrinkage']

    seed_int = 0
    # seed_int=1673588891
    # seed_int = int(time.time())
    np.random.seed(seed_int)
    seed = True

    # print('seed_int:', seed_int, 'combine:', combine, ', shrinkage_spatial:', round(shrinkage_spatial,5), ', score:', round(good_score_list_sort[i_select], 4))

    # statistics_driven
    # fuzzy_logic

    model_params_spectral = {'shrinkage': shrinkage_spectral, 'solver': 'eigen'}
    # model_params_spectral={'n_components': 28,}
    # model_params_spectral=None

    # model_params_spatial={'shrinkage': shrinkage_spatial, 'solver': 'eigen'}
    model_params_spatial = None
    # model_params_spatial={'n_components': 5,}

    # LinearDiscriminantAnalysis

    # 28 5 6 22 1
    # model_params={'n_components': 1,}
    # PLSRegression

    spectral_model = LinearDiscriminantAnalysis
    spatial_model = LinearDiscriminantAnalysis

    # X_norm
    # spatial_features_norm
    # X_norm_with_spatial_features_norm
    # X_norm_with_spatial_features_norm_MRMR_norm
    # X_norm_with_spatial_features_norm_PCA_norm
    # X_norm_mrmr
    # spatial_features_mrmr
    # spatial_features_norm_MRMR

    feature1 = X_norm
    feature2 = spatial_features_norm

    (precision_spectrals, precision_spatials, precision_ensemble, recall_spectrals, recall_spatials, recall_ensemble, accuracy_spectrals, accuracy_spatials, accuracy_ensemble, all_y_train, all_score_spectral_norms_train, all_score_spatial_norms_train, all_pred_spectral_norms_train, all_pred_spatial_norms_train, all_score_spectrals_train, all_score_spatials_train, all_y_test, all_score_spectral_norms_test, all_score_spatial_norms_test,  all_pred_spectral_norms_test, all_pred_spatial_norms_test, all_score_spectrals_test, all_score_spatials_test, all_bias
     ) = repeated_test_ensamble(spectral_model, spatial_model, feature1, feature2, label, test_num=100, seed=seed, model_params_spectral=model_params_spectral, model_params_spatial=model_params_spatial, ensemble_method='statistics_driven')
    show_statistics(precision_ensemble), show_statistics(recall_ensemble), show_statistics(accuracy_ensemble)
     


if __name__ == '__main__':
    main()
