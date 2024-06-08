import random
from itertools import combinations

import torch
# import torch_directml
import torchvision.transforms.functional
from torch import nn
import os
import glob
from typing import List
from dataset_pld_new import DataBaseCBV
from torch.utils.data import DataLoader
# from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from core.models import MultimodalityHalfNetwork as GeneralUNetwork
from core.models import Generic_UNetwork, AdverserialNetwork
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
# from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import time
from core.simulator import Trusteeship, MultipleModalityTrusteeship
import SimpleITK as sitk
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(0)

# input_size = (256, 288, 16)
input_size = (320, 320, 19)
training_data = DataBaseCBV(
    root="E:/dataset/ASL_to_CBV/",
    datafolder="PLD_ASL_Train/need_reslice",
    data_shape=input_size,
    run_stage='train',
    use_augment=False,
    modality_random=False,
    use_realign=False,
    aug_stride=(1, 1, 1),
    aug_side=(32, 32, 32),
    return_original=False,
)

test_data = DataBaseCBV(
    root="E:/dataset/ASL_to_CBV/",
    datafolder="PLD_ASL_Test",
    data_shape=input_size,
    run_stage='test',
    use_augment=False,
    use_realign=True,
    modality_random=False,
    return_original=False,
)

extra_data = [DataBaseCBV(
    root="E:/dataset/ASL_to_CBV/extra_validation",
    datafolder=datafolder,
    data_shape=input_size,
    run_stage='extra',
    use_augment=False,
    return_original=True,
) for datafolder in ['extra_validation2', "Grade&PrognosisE", "Correlation_test"]]



batch_size = 1
train_dataloader = DataLoader(training_data, batch_size=batch_size,)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
extra_dataloader = [DataLoader(extra_data[idx], batch_size=batch_size) for idx in range(len(extra_data))]

# for essamble, subjname in test_dataloader:
#     for ida in essamble:
#         print(f"Shape of {ida} [N, C, H, W]: {essamble[ida].shape}")
#     break
# torch_directml.device_count()
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available GPUs are {torch.cuda.get_device_name(0), torch.cuda.device_count()}")
print(f"Using {device} device")
basedim = 16
# general_model = GeneralUNetwork(1, 16, basedim=8, downdepth=1, unetdepth=3, model_type='2.5D', isresunet=True, use_triD=False, activation_function='tanh')
brainmodalitysynthesis = MultipleModalityTrusteeship(
    nn.ModuleDict({'T1WI': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='lrelu'),
                   'T2WI': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='lrelu'),
                   'T1_C': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='lrelu'),
                   'ADC': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='lrelu'),
                   'FLAIR': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='lrelu'),
                   'CBV': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='relu'),
                   'ASL': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='lrelu'),
                   'brainmask': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='sigmoid'),
                   'Trans': GeneralUNetwork(1, 16, basedim=basedim, downdepth=0, unetdepth=4, model_type='2.5D', isresunet=True, use_triD=False, activation_function='lrelu'),
                   }),
    loss_fn={'T1WI': ('mae', 'mse'), 'T2WI': ('mae', 'mse', ), 'T1_C': ('mae', 'mse', ), 'ADC': ('mae', 'mse', ), 'FLAIR': ('mae', 'mse', ), 'CBV': ('mae', 'mse', ), 'ASL': ('mae', 'mse', ), 'brainmask': ('mae', 'dice')},
    # loss_fn={'T1WI': ('mae', 'adv'), 'T2WI': ('mae',  'adv'), 'T1_C': ('mae',  'adv'), 'ADC': ('mae',  'adv'),
    #          'FLAIR': ('mae', 'adv'), 'CBV': ('mae',  'adv'), 'ASL': ('mae',  'adv')},
    advmodules=nn.ModuleDict({'T1WI': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D', activation_function=None),
                              'T2WI': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D', activation_function=None),
                              'T1_C': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D', activation_function=None),
                              'ADC': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D',  activation_function=None),
                              'FLAIR': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D',  activation_function=None),
                              'CBV': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D', activation_function=None),
                              'ASL': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D', activation_function=None),
                              'Trans': AdverserialNetwork(1, basedim=16, downdepth=3, model_type='2.5D', activation_function=None),
                              }),
    device=device)

brainmodalitysynthesis.volume_names = {'input': 'orig', 'output': 'mask'}


def train(dataloader, mod_in=None, mod_out=None):
    size = len(dataloader.dataset)
    brainmodalitysynthesis.train()

    for batch, (essamble, subj) in enumerate(dataloader):
        if 'CBV' not in essamble: continue
        for it in range(1):
            if 'brainmask' in essamble and random.randint(0, 1) == 1:
                datadict = {it: [(essamble[it] * essamble['brainmask']).to(device), random.choice([0, 1]), 1] for it in essamble}
            else:
                datadict = {it: [essamble[it].to(device), random.choice([0, 1]), 1] for it in essamble}
            if mod_in is not None:
                for md in datadict: datadict[md][1] = 1 if md in mod_in else 0
            if mod_out is not None:
                for md in datadict: datadict[md][2] = 1 if md in mod_out else 0
            if len(datadict) == 0: continue
            # datadict['CBV'][2] = 1
            # datadict['CBV'][1] = 0
            datadict['brainmask'][1] = 0
            loss_ensamble = brainmodalitysynthesis.train_step(datadict)
            loss_total = [loss_ensamble[ss].item() for ss in loss_ensamble]

        if any([np.isnan(ls) for ls in loss_total]):
            print(subj)

        if any([lt > 0.8 for lt in loss_total]):
            print(subj)

        if batch % 2 == 0:
            current = batch * batch_size
            print(f"total loss: {','.join(['%7f' % lt for lt in loss_total])}  [{current:>5d}/{size:>5d}]")


def test1toN(dataloader, mod_in=('T1WI',), mod_out=('T1WI', 'T2WI', 'T1_C', 'FLAIR', 'ADC', 'CBV'), num_modin=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    brainmodalitysynthesis.eval()

    seg_loss_all, correct_psnr_all, correct_ssim_all, correct_rmae_all = [], [], [], []

    with (torch.no_grad()):
        for essamble, subj in dataloader:
            ismissing = [modal not in essamble for modal in mod_out]
            if any(ismissing): continue
            seg_loss, correct_psnr, correct_rmae, correct_ssim = [], [], [], []
            datadict = {it: [essamble[it].to(device), 0, 0] for it in essamble}
            images_all = [torch.concat([datadict[lt][0][0, :, 8, :, :] for lt in mod_out[::-1]], dim=2).detach().cpu().numpy()]

            for md in mod_out: datadict[md][2] = 1
            BM = essamble['brainmask'].numpy()[0,]
            if num_modin == 0: num_modin = len(mod_in)
            for modal in list(combinations(mod_in, num_modin)):
                for md in datadict: datadict[md][1] = 0
                for md in modal: datadict[md][1] = 1
                datadict['brainmask'][2] = 0
                # datadict['CBV'][1] = 0
                loss_ensamble, predictions = brainmodalitysynthesis.eval_step(datadict)
                images_all.append(torch.concat([predictions[lt][1][0, :, 8, :, :] for lt in mod_out[::-1]], dim=2).detach().cpu().numpy())

                seg_loss += [loss_ensamble[lt].item() for lt in mod_out]
                y_pred, y_true = [predictions[lt][1].cpu().numpy()[0,] for lt in mod_out], [datadict[lt][0].cpu().numpy()[0,] for lt in mod_out]
                y_pred = [y_pred[lt]/np.mean(y_pred[lt]*BM)*0.25*np.mean(BM) for lt in range(len(y_pred))]
                correct_psnr += [psnr(y_pred[lt], y_true[lt], data_range=1.0) for lt in range(len(mod_out))]
                correct_rmae += [nrmse(y_pred[lt], y_true[lt]) for lt in range(len(mod_out))]
                correct_ssim += [ssim(y_pred[lt][0], y_true[lt][0], data_range=1.0) for lt in range(len(mod_out))]
                # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj['BID'][0], 'all_' + modal + '_ALL.png'), images_all[-1][0])
            print(subj, f"{'FLAIR'} loss: {','.join(['%7f' % lt for lt in seg_loss])}",
                  f"psnr: {','.join(['%7f' % lt for lt in correct_psnr])}",
                  f"rmae: {','.join(['%7f' % lt for lt in correct_rmae])}",
                  f"ssim: {','.join(['%7f' % lt for lt in correct_ssim])}")

            seg_loss_all.append(seg_loss)
            correct_psnr_all.append(correct_psnr)
            correct_rmae_all.append(correct_rmae)
            correct_ssim_all.append(correct_ssim)

            all_img = np.concatenate(images_all[::-1], axis=1)
            plt.imshow(np.transpose(all_img, axes=(1, 2, 0)))
            # plt.show()
            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'FLAIR'+'.png'), all_img[0])
            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'missone' + '.png'), all_img[0])

        print(f"total segmentation loss: {','.join(['%7f' % lt for lt in np.mean(seg_loss_all, 0)])}")
        print(f"correct_psnr: {','.join(['%7f' % lt for lt in np.mean( correct_psnr_all, 0)])}")
        print(f"correct_rmae: {','.join(['%7f' % lt for lt in np.mean(correct_rmae_all, 0)])}")
        print(f"correct_ssim: {','.join(['%7f' % lt for lt in np.mean( correct_ssim_all, 0)])}")


def test(dataloader, ):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    brainmodalitysynthesis.eval()
    seg_loss_all, correct_psnr_all, correct_ssim_all, correct_rmse_all, correct_rmae_all = [], [], [], [], []

    with (torch.no_grad()):
        modalities = ['T1WI', 'T2WI', 'T1_C', 'FLAIR', 'ADC', 'ASL', 'CBV']
        for essamble, subj in dataloader:
            if 'CBV' not in essamble:
                continue
            test_loss, correct_psnr, correct_mae, correct_rmse, correct_ssim = [], [], [], [], []
            datadict = {it: [essamble[it].to(device), random.choice([0, 0]), 0] for it in essamble}
            datadict['CBV'][2] = 1
            images_all = [torch.concat([datadict[lt][0][0, :, 8, :, :] for lt in datadict if datadict[lt][2] == 1], dim=2).detach().cpu().numpy()]
            in_modalities = ['T1WI', 'T2WI', 'T1_C', 'FLAIR', 'ADC', 'ASL', ]
            # all_combination = [modalcom for cnt in range(len(in_modalities)) for modalcom in list(combinations(in_modalities, cnt+1))]
            all_combination = [('ASL',), ('ASL', 'T2WI'), ('ASL', 'T1WI'), ('ASL', 'FLAIR'), ('ASL', 'ADC'),
                               ('ASL', 'T1WI', 'T2WI'), ('ASL', 'T1WI', 'FLAIR'), ('ASL', 'T1WI', 'ADC'), ('ASL', 'FLAIR', 'ADC'), ('ASL', 'T2WI', 'FLAIR'), ('ASL', 'T2WI', 'ADC'),
                               ('ASL', 'T1WI', 'T2WI', 'FLAIR'), ('ASL', 'T1WI', 'T2WI', 'ADC'), ('ASL', 'T2WI', 'FLAIR', 'ADC'), ('ASL', 'T1WI', 'FLAIR', 'ADC'),
                               ('ASL', 'T1WI', 'T2WI', 'FLAIR', 'ADC'),]
            for modalcom in all_combination:
                for md in datadict:
                    datadict[md][1] = 1 if md in modalcom else 0

                loss_ensamble, predictions = brainmodalitysynthesis.eval_step(datadict)
                # for itx in range(3):
                #     datadict_new = datadict.copy()
                #     datadict_new['CBV'] = [predictions['CBV'][1], 1, 0]
                #     loss_ensamble, predictions = brainmodalitysynthesis.eval_step(datadict_new)

                images_all.append(torch.concat([predictions[lt][1][0, :, 8, :, :] for lt in datadict if datadict[lt][2] == 1], dim=2).detach().cpu().numpy())
                test_loss += [loss_ensamble[lt].item() for lt in datadict if datadict[lt][2] == 1]
                BM = essamble['brainmask'].numpy()[0,]
                y_pred, y_true = [predictions[lt][1].cpu().numpy()[0,]*BM for lt in datadict if datadict[lt][2] == 1], [datadict[lt][0].cpu().numpy()[0,]*BM for lt in datadict if datadict[lt][2] == 1]

                y_pred = [y_pred[lt]/np.mean(y_pred[lt]*BM)*0.25*np.mean(BM) for lt in range(len(y_pred))]
                correct_mae += [np.mean(np.abs(y_pred[lt]-y_true[lt])) for lt in range(len(y_pred))]
                correct_psnr += [psnr(y_pred[lt], y_true[lt], data_range=1.0) for lt in range(len(y_pred))]
                correct_rmse += [np.sqrt(mse(y_pred[lt], y_true[lt])) for lt in range(len(y_pred))]
                correct_ssim += [ssim(y_pred[lt][0], y_true[lt][0], data_range=1.0) for lt in range(len(y_pred))]

                # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj['BID'][0], 'all_' + modal + '_ALL.png'), images_all[-1][0])
            print(subj, #f"{'CBV'} loss: {','.join(['%2.6f' % lt for lt in test_loss])}",
                  f"psnr: {','.join(['%2.6f' % lt for lt in correct_psnr])}",
                  f"rmae: {','.join(['%2.6f' % lt for lt in correct_mae])}",
                  f"rmse: {','.join(['%2.6f' % lt for lt in correct_rmse])}",
                  f"ssim: {','.join(['%2.6f' % lt for lt in correct_ssim])}")
            seg_loss_all.append(test_loss)
            correct_psnr_all.append(correct_psnr)
            correct_rmae_all.append(correct_mae)
            correct_rmse_all.append(correct_rmse)
            correct_ssim_all.append(correct_ssim)
            # all_img = np.concatenate([np.concatenate(images_all[idx::8], axis=1) for idx in range(8)], axis=2)
            # plt.imshow(np.transpose(all_img, axes=(1, 2, 0)), vmin=0, vmax=1.0,)
            #pass
            # break
            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'FLAIR'+'.png'), all_img[0])
            # plt.imsave(os.path.join("D:\dataset/A4/A4_SYN", subj[0], 'missone' + '.png'), all_img[0])


    for lt in range(len(all_combination)):
            print(all_combination[lt])
            print("total loss:%6f\261%7f,"% (np.mean(seg_loss_all, 0)[lt], np.std(seg_loss_all, 0)[lt]),
                  "psnr:%6f\261%7f,"% (np.mean(correct_psnr_all, 0)[lt], np.std(correct_psnr_all, 0)[lt]),
                  "rmae:%6f\261%7f,"% (np.mean(correct_rmae_all, 0)[lt], np.std(correct_rmae_all, 0)[lt]),
                  "rmse:%6f\261%7f," % (np.mean(correct_rmse_all, 0)[lt], np.std(correct_rmse_all, 0)[lt]),
                  "ssim:%6f\261%7f," % (np.mean(correct_ssim_all, 0)[lt], np.std(correct_ssim_all, 0)[lt]))

    # print(f"total loss: {','.join(['%7f' % lt for lt in np.mean(seg_loss_all, 0)])}")
    # print(f"correct_psnr: {','.join(['%7f' % lt for lt in np.mean( correct_psnr_all, 0)])}")
    # print(f"correct_rmae: {','.join(['%7f' % lt for lt in np.mean(correct_rmae_all, 0)])}")
    # print(f"correct_rmse: {','.join(['%7f' % lt for lt in np.mean(correct_rmse_all, 0)])}")
    # print(f"correct_ssim: {','.join(['%7f' % lt for lt in np.mean( correct_ssim_all, 0)])}")


def apply(dataloader, mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC',), mod_out=('CBV',), num_modin=1):
    brainmodalitysynthesis.eval()
    dataloader.dataset.return_original = True

    with (torch.no_grad()):
        for essamble, subjinfo in dataloader:
            datadict = {it: [essamble[it].to(device), random.choice([0, 0]), 0] for it in essamble}
            subj = subjinfo[0]
            tarpath = os.path.join(dataloader.dataset.root, dataloader.dataset.datafolder+'SYN', subj)
            if not os.path.exists(tarpath): os.makedirs(tarpath)

            for modalcom in list(combinations(mod_in, num_modin)):
                for md in datadict: datadict[md][1] = 1 if md in modalcom else 0
                if sum([datadict[it][1] for it in datadict]) == 0: continue

                predictions = brainmodalitysynthesis.infer_step(datadict)
                datadict_new = datadict.copy()
                datadict_new['CBV'] = [predictions['CBV'][1], 1, 0]
                predictions = brainmodalitysynthesis.infer_step(datadict_new)

                ref = [modal for modal in datadict if datadict[modal][1] == 1][0]
                ref_img = sitk.ReadImage(os.path.join(dataloader.dataset.processed_folder, subj, ref+'n.nii.gz'))
                for modal in predictions:
                    if modal not in mod_out: continue
                    if modal in ['brainmask',]:
                        modal_itk = sitk.GetImageFromArray(np.int16(predictions[modal][1][0, 0].cpu().numpy()>0.5))
                        modal_itk = sitk.BinaryMorphologicalClosing(modal_itk, kernelRadius=[5, 5, 1])
                    else:
                        modal_itk = sitk.GetImageFromArray(np.int16(1000*torch.relu(predictions[modal][1][0, 0]).cpu().numpy()))
                    modal_itk.SetOrigin(ref_img.GetOrigin())
                    modal_itk.SetSpacing(ref_img.GetSpacing())
                    modal_itk.SetDirection(ref_img.GetDirection())
                    sitk.WriteImage(modal_itk, os.path.join(tarpath, '{0}_sb_{1}.nii.gz'.format(modal, ''.join(modalcom))), useCompression=True)
            # plt.imshow(np.transpose(torch.cat([ess_synth[modal][0, :, 127, :, :] for modal in ess_synth], dim=2).detach().cpu().numpy(), axes=(1, 2, 0)))
            # # plt.show()
            # pass


epochs = 4001
start_epochs = 1600
if start_epochs > 0:
    for modal in brainmodalitysynthesis.modules: #['T1WI', 'T2WI', 'T1_C', 'ADC', 'FLAIR', 'CBV', 'ASL', 'Trans']:
        # if modal == 'brainmask': continue
        brainmodalitysynthesis.load_dict(modal, start_epochs, prefix='chkpt', strict=False)

for t in range(start_epochs, epochs):
    print(f"Epoch {t}\n-------------------------------")
    start_time = time.time()
    # train(train_dataloader, mod_out=('T1WI', 'CBV', 'brainmask'))
    # test1toN(train_dataloader, mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC',), mod_out=('CBV',), num_modin=1) #'T1WI', 'T2WI', 'FLAIR', 'ADC', 'CBV'
    test(test_dataloader, )
    # for idx in range(len(extra_dataloader)):
    #     apply(extra_dataloader[idx], mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC', 'ASL'), mod_out=('CBV'), num_modin=1)
    #     apply(extra_dataloader[idx], mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC', 'ASL'), mod_out=('CBV',), num_modin=5)
    #     apply(extra_dataloader[idx], mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC', 'ASL'), mod_out=('CBV',), num_modin=6)

    end_time = time.time()
    print(f"Epoch {t} takes {end_time-start_time} seconds")
    break
    if t % 100 == 99:
        for modal in brainmodalitysynthesis.modules: #['T1WI', 'T2WI', 'T1_C', 'ADC', 'FLAIR', 'CBV', 'ASL', 'Trans']:
            brainmodalitysynthesis.save_dict(modal, t+1, prefix='chkpt')

print("Done!")

# wasl_files = glob.glob('E:\dataset\ASL_to_CBV\PLD_ASL_Test\*\*\T1_C.nii.gz')
# woasl_files = glob.glob('E:\dataset\ASL_to_CBV\PLD_ASL_Train\\need_reslice\*\*\T1_C.nii.gz')
#
#
# for wfile in wasl_files:
#     wimg = sitk.GetArrayFromImage(sitk.ReadImage(wfile))
#     # print(os.stat(wfile))
#     for wofile in woasl_files:
#         woimg = sitk.GetArrayFromImage(sitk.ReadImage(wofile))
#         # time.sleep(0.1)
#         if np.shape(woimg) == np.shape(wimg):
#             if np.mean(np.abs(woimg-wimg)) == 0:
#                 print(wfile, wofile)
#                 for aslfl in ['TGSE.nii.gz', 'ASL.nii.gz', 'DWI.nii.gz']:
#                     if os.path.exists(os.path.join(os.path.dirname(wfile), aslfl)):
#                         shutil.copy(os.path.join(os.path.dirname(wfile), aslfl), os.path.join(os.path.dirname(wofile), aslfl))
#                         shutil.rmtree(os.path.dirname(wfile))

