"""
 @file   01_test.py
 @brief  Script for test
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common_base as com
import ares_torch_model
import eval_functions_eembc
import pylab
import torch

########################################################################
# import custom libraries
########################################################################
import utils
import plot_utils
import arg_parser

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load("baseline_ares.yaml")
#param = arg_parser()
#######################################################################


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    print(mode)
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    print("============== PARAMETERS ==============")
    for key, val in param.items():
        print("%s = %s" %(key, str(val)))

    print("============== DIRECTORIES ==============")
    print("%s" %str(dirs))

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # set model path
        model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                machine_type=machine_type)

        print("model directory = %s" %str(param["model_directory"]))
        # load model file
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)

        batch_size = param["fit"]["batch_size"]
        epochs = param["fit"]["epochs"]

        model = ares_torch_model.noisy_autoencoder(param)
        model_file_path = "{model}/model_ToyCar.hdf5".format(model=param["model_directory"])
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        criterion = checkpoint['criterion']
        loss_history = checkpoint['loss_history']
        start_epoch = len(loss_history) + 1 #checkpoint['epoch']

        #model = ares_torch_model.noisy_autoencoder(param)
        #model.load_state_dict(ares_torch_model.load_model(model_file))
        model.eval()

        print("============== LAYER CONFIGS ==============")
        for name, data in model.named_parameters():

            extracted_dir = param["model_directory"] + '/extracted_model'
            activations_dir = param["model_directory"] + '/activations'
            os.makedirs(extracted_dir, exist_ok=True)
            os.makedirs(activations_dir, exist_ok=True)

            data = data.detach().cpu().numpy()

            if len(data.shape) == 2:
                #print(weights)
                #print(type(weights))
                weights_path = extracted_dir + '/' + name + '_weights_' + str(data.shape).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '') + '.csv'
                np.savetxt(weights_path, data, fmt='%f', delimiter=',')

            if len(data.shape) == 1:
                biases_path = extracted_dir + '/' + name + '_biases_' + str(data.shape).replace('(', '').replace(')', '').replace(',', '').replace(' ', '') + '.csv'
                np.savetxt(biases_path, data, fmt='%f', delimiter=',')

            #print(layer.get_weights())
        #model.summary()

        if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = com.get_machine_id_list_for_test(target_dir)
        print("Machine ID List: %s" %str(machine_id_list))

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = com.test_file_list_generator(target_dir, id_str, mode)

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                     result=param["result_directory"],
                                                                                     machine_type=machine_type,
                                                                                     id_str=id_str)
            anomaly_score_list = []

            print("\n============== BEGIN TEST FOR A MACHINE ID: %d ==============" %idx)
            y_pred = [0. for k in test_files]
            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                #print("\nfile_path: %s" %file_path)
                try:
                    data = com.file_to_vector_array(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    frames=param["feature"]["frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    win_length=param["feature"]["win_length"],
                                                    power=param["feature"]["power"],
                                                    save_png=False)
                    #print("input data dimensions = %s" %str(data.shape))
                    #pred = model.predict(data)
                    pred = model(torch.tensor(data, dtype=torch.float32))

                    #print("output data dimensions = %s" %str(pred.shape))

                    # pylab.figure()
                    # pylab.imshow(data, interpolation='nearest')
                    # pylab.savefig('input_' + str(file_idx) + '.png')
                    # pylab.close()
                    #
                    # pylab.figure()
                    # pylab.imshow(pred, interpolation='nearest')
                    # pylab.savefig('output_' + str(file_idx) + '.png')
                    # pylab.close()
                    # sys.exit()

                    pred = pred.detach().cpu().numpy()
                    y_pred[file_idx] = np.mean(np.square(data - pred))
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

                    label = 'anomaly' if 'anomaly' in file_path.split('/')[-1] else 'normal'
                    input_activations_path = activations_dir + '/input_' + id_str + '_idx_' + str(file_idx) + '_' + label + '_shape_' + str(pred.shape).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '') + '.csv'
                    output_activations_path = activations_dir + '/output_' + id_str + '_idx_' + str(file_idx) + '_' + label + '_shape_' + str(pred.shape).replace('(', '').replace(')', '').replace(',', '_').replace(' ', '') + '.csv'
                    np.savetxt(input_activations_path, data, fmt='%f', delimiter=',')
                    np.savetxt(output_activations_path, pred, fmt='%f', delimiter=',')

                except Exception as e:
                    com.logger.error("file broken!!: {}, {}".format(file_path, e))

            # save anomaly score
            com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))

                # ROC Curve
                y_pred_np = np.asarray(y_pred)
                y_true_np = np.asarray(y_true)
                TPR_list, FPR_list = [], []

                inc = 0.01
                thresholds = np.arange(np.amin(y_pred_np)+inc, np.amax(y_pred_np)-inc, inc)
                for thres in thresholds:
                    anom = np.where(y_pred_np > thres, 1, 0)

                    TP = np.sum(anom * y_true_np)                           # where both pos
                    TN = np.sum(np.where(anom + y_true_np < 1, 1, 0))  # where both zero

                    FN = np.sum(np.where(anom < y_true_np, 1, 0))      # where pred < true
                    FP = np.sum(np.where(anom > y_true_np, 1, 0))      # where pred > true

                    TPR = TP / (TP + FN)    # true pos rate
                    FPR = FP / (FP + TN)    # false pos rate

                    TPR_list.append(TPR)
                    FPR_list.append(FPR)

                true_pos_rate = np.asarray(TPR_list)
                false_pos_rate = np.asarray(FPR_list)

                roc_auc = np.abs(np.trapz(true_pos_rate, x=false_pos_rate)) # abs because order

                np.savetxt(param["result_directory"] + '/ROC_data_' + id_str + '.txt', np.asarray([FPR_list, TPR_list]).T, fmt='%f', delimiter=',')

                x_list = [false_pos_rate, false_pos_rate]
                y_list = [true_pos_rate, false_pos_rate]
                legend_list = ["ROC (AUC = %0.2f)" %roc_auc, 'REF']
                plot_utils.plot_1d_overlay(x_list,
                                           y_list,
                                           LEGEND_LIST=legend_list,
                                           XLABEL='False Positive Rate',
                                           YLABEL='True Positive Rate',
                                           SAVE_NAME=param["result_directory"] + '/ROC_' + id_str,
                                           markers=False
                                           )

                acc_eembc = eval_functions_eembc.calculate_ae_accuracy(y_pred, y_true)
                pr_acc_eembc = eval_functions_eembc.calculate_ae_pr_accuracy(y_pred, y_true)
                auc_eembc = eval_functions_eembc.calculate_ae_auc(y_pred, y_true, "dummy")
                com.logger.info("EEMBC Accuracy: {}".format(acc_eembc))
                com.logger.info("EEMBC Precision/recall accuracy: {}".format(pr_acc_eembc))
                com.logger.info("EEMBC AUC: {}".format(auc_eembc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        com.save_csv(save_file_path=result_path, save_data=csv_lines)
