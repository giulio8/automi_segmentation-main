'''
This script is used to evaluate the results of the segmentation models that were saved as .npz files.
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils_evaluation
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_folder', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--evaluation_path', type=str, required=True)
    parser.add_argument('--structure', type=str, required=True)
    args = parser.parse_args()

    evaluationSet = 'AUTOMI_test'
    folds = [1, 2, 3, 4, 5]
    for fold in folds:
        predictionFolder = os.path.join(args.prediction_folder, evaluationSet, f'fold{fold}predictionsNPZ_noBones')
        evaluationPath = os.path.join(args.prediction_folder, evaluationSet, f'fold{fold}evaluation_noBones.xlsx')

        utils_evaluation.evaluate_folder_npz(predictionFolder, args.img_path, evaluationPath, args.structure)

