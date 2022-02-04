import os
import plot_utils
import numpy as np

def export_torch_layer_info(model, param):

    dir_name = param["model_name"].replace('.pt', '')
    os.makedirs(param["model_dir"] + '/pdfs_' + dir_name, exist_ok=True)
    i, w, b = 0, 0, 0
    for name, data in model.named_parameters():
        print(name)
        #print(data)
        data = data.detach().cpu().numpy()
        print("layer[%d].shape = %s" %(i, str(data.shape)))
        if len(data.shape) == 1:
            print("Layer %d: max(|b|) = %f" % (i, np.amax(np.abs(data))))
            plot_utils.plot_pdf(data, XLABEL='Biases [1]', YLABEL='PDF [1]',
                                SAVE_NAME=param["model_dir"] + '/pdfs_' + dir_name + '/biases_pdf_' + str(b))
            b_sorted, b_ecdf = plot_utils.get_ecdf(data)
            plot_utils.plot_1d(b_sorted, b_ecdf, XLABEL='Biases [1]', YLABEL='CDF [1]',
                               SAVE_NAME=param["model_dir"] + '/pdfs_'  + dir_name + '/biases_cdf_' + str(b))

            #plot_utils.heatmap(data[np.newaxis,:], XLABEL='Columns', YLABEL='Rows',
            #                   SAVE_NAME=param["model_directory"] + '/pdfs/biases_heatmap_' + str(b))
            b += 1
        elif len(data.shape) == 2:
            print("Layer %d: max(|w|) = %f" % (i, np.amax(np.abs(data))))
            plot_utils.plot_pdf(data, XLABEL='Weights [1]', YLABEL='PDF [1]',
                                SAVE_NAME=param["model_dir"] + '/pdfs_' + dir_name + '/weights_pdf_' + str(w))
            w_sorted, w_ecdf = plot_utils.get_ecdf(data)
            plot_utils.plot_1d(w_sorted, w_ecdf, XLABEL='Weights [1]', YLABEL='CDF [1]',
                               SAVE_NAME=param["model_dir"] + '/pdfs_' + dir_name + '/weights_cdf_' + str(w))
            #plot_utils.heatmap(data, XLABEL='Columns', YLABEL='Rows',
            #                   SAVE_NAME=param["model_dir"] + '/pdfs_' + param["model_name"] + '/weights_heatmap_' + str(w))
            w += 1
        i += 1