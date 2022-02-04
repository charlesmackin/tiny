"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys

import numpy as np

import utils
########################################################################


########################################################################
# import additional python-library
########################################################################
import sys
import numpy
import plot_utils
from sklearn import metrics

# from import
from tqdm import tqdm
# original lib
import common as com
import ares_torch_model
import torch
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load("baseline_ares.yaml")
print(param)
########################################################################

def roc_auc(y_pred_np, y_true_np):

    TPR_list, FPR_list = [], []

    inc = 0.05
    thresholds = numpy.arange(numpy.amin(y_pred_np) + inc, numpy.amax(y_pred_np) - inc, inc)
    for thres in thresholds:
        anom = numpy.where(y_pred_np > thres, 1, 0)

        TP = numpy.sum(anom * y_true_np)  # where both pos
        TN = numpy.sum(numpy.where(anom + y_true_np < 1, 1, 0))  # where both zero

        FN = numpy.sum(numpy.where(anom < y_true_np, 1, 0))  # where pred < true
        FP = numpy.sum(numpy.where(anom > y_true_np, 1, 0))  # where pred > true

        TPR = TP / (TP + FN)  # true pos rate
        FPR = FP / (FP + TN)  # false pos rate

        TPR_list.append(TPR)
        FPR_list.append(FPR)

    true_pos_rate = numpy.asarray(TPR_list)
    false_pos_rate = numpy.asarray(FPR_list)

    roc_auc_score = numpy.abs(numpy.trapz(true_pos_rate, x=false_pos_rate))  # abs because order
    return torch.tensor(roc_auc_score, dtype=torch.float32, requires_grad=True)
def list_to_vector_array(file_list, msg="calc...", n_mels=64, frames=5, n_fft=1024, hop_length=512, win_length=None, power=2.0, log_mel_start_ind=50, log_mel_stop_ind=250):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                win_length=win_length,
                                                log_mel_start_ind=log_mel_start_ind,
                                                log_mel_stop_ind=log_mel_stop_ind,
                                                power=power,
                                                save_png=False)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset
def file_list_generator(target_dir, dir_name="train", ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files

########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)
    print(param["model_directory"])

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        print("============== LOAD/MAKE MODEL ==============")

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            #continue

        batch_size = param["fit"]["batch_size"]
        epochs = param["fit"]["epochs"]

        model = ares_torch_model.noisy_autoencoder(param)
        if param["feature"]["resume_from_checkpoint"]:
            print("Training from checkpoint: %s" %model_file_path)
            checkpoint = torch.load(model_file_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            criterion = checkpoint['criterion']
            loss_history = checkpoint['loss_history']
            valid_loss_history = checkpoint['valid_loss_history']
            start_epoch = len(loss_history) + 1 #checkpoint['epoch']
            if start_epoch > epochs:
                sys.exit('start_epoch > yaml epochs: will not run anything')
        else:
            print("Training from scratch: %s" %model_file_path)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)
            model.to(device)
            criterion = torch.nn.MSELoss()
            #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.1, weight_decay=0.0001)
            #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

            # same as default keras
            # optimizer = torch.optim.Adam(model.parameters(),
            #                              lr=0.001,
            #                              betas=(0.9, 0.999),
            #                              eps=1e-07,
            #                              amsgrad=False,
            #                              weight_decay=0.0001)

            # same as default keras
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=0.001,
                                         betas=(0.9, 0.999),
                                         eps=1e-07,
                                         amsgrad=False,
                                         weight_decay=param["feature"]["l2_reg"])

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=1.0)
            loss_history = []
            valid_loss_history = []
            start_epoch = 1

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir)
        training_data = list_to_vector_array(files,
                                             msg="generate train_dataset",
                                             n_mels=param["feature"]["n_mels"],
                                             frames=param["feature"]["frames"],
                                             n_fft=param["feature"]["n_fft"],
                                             hop_length=param["feature"]["hop_length"],
                                             win_length=param["feature"]["win_length"],
                                             power=param["feature"]["power"],
                                             log_mel_start_ind=param["feature"]["log_mel_start_ind"],
                                             log_mel_stop_ind=param["feature"]["log_mel_stop_ind"]
                                             )
        print("train_data.shape = %s" %str(training_data.shape))
        training_data = torch.tensor(training_data, dtype=torch.float32)
        #training_data = training_data.to(device)

        # train model
        print("============== MODEL TRAINING ==============")
        model.train()
        for epoch in range(start_epoch, epochs+1, 1):

            training_data = training_data[torch.randperm(training_data.size()[0])]   # shuffle rows

            vfrac = param["fit"]["validation_split"]
            valid_data = training_data[0:int(training_data.shape[0]*vfrac), :]
            train_data = training_data[int(vfrac*training_data.shape[0]):, :]

            total_loss = 0
            for batch_idx in range(int(train_data.shape[0] / batch_size)):

                ind1 = batch_size * batch_idx
                ind2 = batch_size * batch_idx + batch_size - 1
                ind2 = train_data.shape[0] if ind2 > train_data.shape[0] else ind2

                input_data = train_data[ind1:ind2, :].to(device)
                y_pred = model(input_data)

                loss = criterion(y_pred, input_data)
                #test_files, y_true = com.test_file_list_generator(target_dir, id_str, mode)
                #loss = metrics.roc_auc_score(y_true, y_pred.detach().cpu().numpy()) if bool(param["auc_train"]) else criterion(y_pred, input_data)
                #loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
                #loss = roc_auc(input_data.detach().cpu().numpy(), y_pred.detach().cpu().numpy()) if bool(param["auc_train"]) else criterion(y_pred, input_data)
                #loss = torch.tensor(loss) if bool(param["auc_train"]) else loss

                total_loss = total_loss + loss * (ind2-ind1)
                if batch_idx % 200 == 0:
                    print("Epoch: %d, Batch_idx: %d, Loss: %f (%0.1f %%)" %(epoch, batch_idx, total_loss/ind2, 100*ind1/train_data.shape[0]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Clip Gradients
                #torch.nn.utils.clip_grad_norm_(model.parameters(), param["max_grad_norm"])

                # Clip Weights
                for name, dummy in model.named_parameters():
                    dummy.data = torch.clamp(dummy.data, min=-param["max_weight"], max=param["max_weight"])

            utils.export_torch_layer_info(model, param)
            torch.save({'epoch': epoch,
                        'loss': loss,
                        'loss_history': loss_history,
                        'valid_loss_history': valid_loss_history,
                        'model_state_dict': model.state_dict(),
                        'criterion': criterion,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        },
                       model_file_path)

            valid_pred = model(valid_data)
            valid_loss = criterion(valid_pred, valid_data)
            valid_loss_history.append(valid_loss)
            #if valid_loss > total_loss / train_data.shape[0]:   # not improving

            print("\n")
            loss_history.append(total_loss / train_data.shape[0])

            # plot_utils.plot_loglog(np.arange(1, len(loss_history) + 1, 1),
            #                        np.asarray(loss_history),
            #                        XLABEL='Epoch',
            #                        YLABEL='Loss',
            #                        SAVE_NAME=history_img)

            plot_utils.plot_loglog_overlay([np.arange(1, len(loss_history) + 1, 1)]*2,
                                           [np.asarray(loss_history), np.asarray(valid_loss_history)],
                                           XLABEL='Epoch',
                                           YLABEL='Loss',
                                           LEGEND_LIST=['Train', 'Valid'],
                                           SAVE_NAME=history_img)

        #model.summary()
        #model.compile(**param["fit"]["compile"])

        # plot_utils.plot_loglog(np.arange(1, len(loss_history) + 1, 1),
        #                        np.asarray(loss_history),
        #                        XLABEL='Epoch',
        #                        YLABEL='Loss',
        #                        SAVE_NAME=history_img + '_complete')

        plot_utils.plot_loglog_overlay([np.arange(1, len(loss_history) + 1, 1)] * 2,
                                       [np.asarray(loss_history), np.asarray(valid_loss_history)],
                                       XLABEL='Epoch',
                                       YLABEL='Loss',
                                       LEGEND_LIST=['Train', 'Valid'],
                                       SAVE_NAME=history_img)

        utils.export_torch_layer_info(model, param)

        torch.save({'epoch': epoch,
                    'loss': loss,
                    'loss_history': loss_history,
                    'valid_loss_history': valid_loss_history,
                    'model_state_dict': model.state_dict(),
                    'criterion': criterion,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    },
                   model_file_path)

        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")
