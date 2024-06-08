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

input_size = (256, 288, 16)
# input_size = (320, 320, 19)
training_data = DataBaseCBV(
    root="E:/dataset/ASL_to_CBV/",
    datafolder="PLD_ASL_Train/need_reslice",
    data_shape=input_size,
    run_stage='train',
    use_augment=False,
    use_realign=False,
    modality_random=False,
    aug_stride=(1, 1, 1),
    aug_side=(32, 32, 32),
    return_original=True,
)

test_data = DataBaseCBV(
    root="E:/dataset/ASL_to_CBV/",
    datafolder="PLD_ASL_Test",
    data_shape=input_size,
    run_stage='test',
    use_augment=False,
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
# ) for datafolder in ['extra_validation2', "Grade&PrognosisE", "Correlation_test"]]
) for datafolder in ["illustration", ]]

batch_size = 1
train_dataloader = DataLoader(training_data, batch_size=batch_size,)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
extra_dataloader = [DataLoader(extra_data[idx], batch_size=batch_size) for idx in range(len(extra_data))]


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


print(f"Epoch {start_epochs}\n-------------------------------")
start_time = time.time()
    # train(train_dataloader, mod_out=('T1WI', 'CBV', 'brainmask'))
    # test1toN(train_dataloader, mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC',), mod_out=('CBV',), num_modin=1) #'T1WI', 'T2WI', 'FLAIR', 'ADC', 'CBV'
    # test(test_dataloader, )

for idx in range(len(extra_dataloader)):
    # apply(extra_dataloader[idx], mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC', 'ASL', 'CBV'), mod_out=('T1WI', 'brainmask'), num_modin=1)
    apply(extra_dataloader[idx], mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC', 'ASL'), mod_out=('CBV',), num_modin=2)
    apply(extra_dataloader[idx], mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC', 'ASL'), mod_out=('CBV',), num_modin=3)
    apply(extra_dataloader[idx], mod_in=('T1WI', 'T2WI', 'FLAIR', 'T1_C', 'ADC', 'ASL'), mod_out=('CBV',), num_modin=4)
end_time = time.time()
print(f"Epoch {start_epochs} takes {end_time-start_time} seconds")



# wasl_files = glob.glob('E:\dataset\ASL_to_CBV\PLD_ASL_Test\*\*\T1_C.nii.gz')
# woasl_files = glob.glob('E:\dataset\ASL_to_CBV\PLD_ASL_Train\\need_reslice\*\*\T1_C.nii.gz')
#
#
# correct_ssim: 0.784939,0.795417,0.809481,0.999166,0.782938,0.999166,0.800633,0.814695,0.784939,0.793388,0.784939,0.814479,0.795417,0.801929,0.795417,0.809481,0.815484,0.809481,0.782938,0.999166,0.782938,0.816374,0.800633,0.804771,0.800633,0.814695,0.817177,0.814695,0.793388,0.784939,0.793388,0.814479,0.817225,0.814479,0.801929,0.795417,0.801929,0.815484,0.809481,0.815484,0.782938,0.816374,0.817855,0.816374,0.804771,0.800633,0.804771,0.817177,0.814695,0.817177,0.793388,0.817225,0.814479,0.817225,0.801929,0.815484,0.817855,0.816374,0.817855,0.804771,0.817177,0.817225,0.817855
# correct_ssim: 0.752015,0.753800,0.792446,0.749865,0.765752,0.998845,0.764563,0.797588,0.760242,0.774583,0.752015,0.795741,0.764605,0.778439,0.753800,0.795870,0.800595,0.792446,0.773193,0.749865,0.765752,0.797560,0.768559,0.782111,0.764563,0.797363,0.802640,0.797588,0.776098,0.760242,0.774583,0.796666,0.801182,0.795741,0.780556,0.764605,0.778439,0.801052,0.795870,0.800595,0.773193,0.797122,0.802249,0.797560,0.782125,0.768559,0.782111,0.801962,0.797363,0.802640,0.776098,0.801236,0.796666,0.801182,0.780556,0.801052,0.801590,0.797122,0.802249,0.782125,0.801962,0.801236,0.801590
# correct_ssim: 0.783603,0.792340,0.809349,0.780236,0.781990,0.999486,0.799213,0.814715,0.790015,0.792658,0.783603,0.814471,0.798718,0.801107,0.792340,0.813602,0.815704,0.809349,0.791735,0.780236,0.781990,0.816262,0.801666,0.804398,0.799213,0.815417,0.817427,0.814715,0.795592,0.790015,0.792658,0.815958,0.817306,0.814471,0.803822,0.798718,0.801107,0.816561,0.813602,0.815704,0.791735,0.816417,0.817972,0.816262,0.805172,0.801666,0.804398,0.817293,0.815417,0.817427,0.795592,0.817525,0.815958,0.817306,0.803822,0.816561,0.817780,0.816417,0.817972,0.805172,0.817293,0.817525,0.817780
# correct_ssim: 0.781430,0.786324,0.800443,0.776972,0.786834,0.999352,0.795888,0.808760,0.787625,0.796042,0.781430,0.808068,0.794342,0.802402,0.786324,0.807159,0.812699,0.800443,0.794631,0.776972,0.786834,0.811371,0.798319,0.805975,0.795888,0.810625,0.815537,0.808760,0.797868,0.787625,0.796042,0.810436,0.814777,0.808068,0.804487,0.794342,0.802402,0.814223,0.807159,0.812699,0.794631,0.812009,0.816214,0.811371,0.805986,0.798319,0.805975,0.815770,0.810625,0.815537,0.797868,0.815394,0.810436,0.814777,0.804487,0.814223,0.816193,0.812009,0.816214,0.805986,0.815770,0.815394,0.816193


# correct_psnr: 18.724387,18.882961,19.118321,18.718389,18.965651,17.387433,19.065565,19.274555,18.861677,19.072487,18.740050,19.276503,19.046881,19.254502,18.850020,19.271637,19.449089,19.190148,19.073975,18.717436,18.943111,19.322762,19.106921,19.288989,19.073921,19.320791,19.460481,19.304851,19.096275,18.900848,19.077930,19.323129,19.458714,19.298020,19.269395,19.059455,19.239969,19.457820,19.299664,19.461963,19.073980,19.351558,19.460627,19.344602,19.280323,19.128774,19.286487,19.456794,19.339609,19.471277,19.115280,19.461755,19.342412,19.471108,19.267258,19.465205,19.461632,19.368724,19.474107,19.285480,19.466475,19.471112,19.470489
# correct_rmae: 0.121387,0.119561,0.116018,0.121366,0.117981,0.139432,0.117232,0.114417,0.119633,0.116872,0.121047,0.114481,0.117415,0.114671,0.119690,0.114419,0.112293,0.115181,0.116803,0.121223,0.118263,0.114025,0.116693,0.114362,0.117049,0.113976,0.112265,0.114074,0.116621,0.119090,0.116752,0.114005,0.112321,0.114178,0.114549,0.117176,0.114819,0.112270,0.114104,0.112169,0.116777,0.113726,0.112364,0.113754,0.114475,0.116383,0.114357,0.112361,0.113763,0.112153,0.116370,0.112322,0.113765,0.112168,0.114555,0.112199,0.112375,0.113507,0.112199,0.114400,0.112255,0.112209,0.112266
# correct_ssim: 0.758705,0.762882,0.778091,0.754008,0.769261,0.671142,0.774388,0.786978,0.765644,0.777496,0.758608,0.785503,0.772570,0.783132,0.761895,0.785177,0.792450,0.780216,0.775577,0.754399,0.767077,0.789242,0.776685,0.786104,0.774508,0.788919,0.794743,0.787520,0.778411,0.767012,0.777195,0.788061,0.793590,0.786370,0.784333,0.772957,0.781817,0.793202,0.785831,0.792339,0.775413,0.790072,0.794866,0.789680,0.785819,0.777367,0.785673,0.794730,0.789292,0.794767,0.778833,0.794016,0.788669,0.793662,0.783999,0.793189,0.794955,0.790558,0.795047,0.785868,0.794849,0.794180,0.795050


# total segmentation loss: 0.238228,0.241431,0.217815,0.237114,0.226285,0.262295,0.233058,0.215983,0.234572,0.224386,0.234412,0.217183,0.232469,0.222833,0.238787,0.215901,0.211340,0.216178,0.224203,0.233563,0.224531,0.216779,0.230590,0.221933,0.230567,0.216526,0.211579,0.215093,0.224731,0.231619,0.222861,0.216689,0.211951,0.216123,0.222157,0.229862,0.221428,0.211614,0.214982,0.210525,0.222623,0.217162,0.212336,0.215859,0.222658,0.228532,0.220780,0.212459,0.215736,0.210883,0.223092,0.212407,0.215835,0.211170,0.220940,0.210877,0.213139,0.216408,0.211681,0.221417,0.211801,0.211734,0.212555
# correct_psnr: 18.256894,18.147922,19.323033,18.330046,18.869146,17.378102,18.524250,19.391359,18.434452,18.890351,18.406155,19.353867,18.578542,19.038909,18.241644,19.407943,19.649082,19.390406,18.913313,18.468382,18.904783,19.353374,18.652292,19.045111,18.600936,19.365504,19.613685,19.403250,18.863017,18.541029,18.922458,19.376124,19.609383,19.380312,19.046375,18.659244,19.056007,19.620357,19.429654,19.662856,18.955562,19.341109,19.574037,19.367914,19.006759,18.713146,19.060970,19.566287,19.373342,19.618679,18.910349,19.582072,19.389245,19.621555,19.066632,19.630180,19.534658,19.348547,19.579991,19.031650,19.569218,19.588104,19.536214
# correct_rmae: 0.126983,0.130266,0.113411,0.125975,0.119041,0.140755,0.123782,0.112657,0.124612,0.118709,0.125163,0.113342,0.123119,0.117238,0.128863,0.112489,0.109734,0.112701,0.118444,0.124270,0.118663,0.113230,0.121949,0.116909,0.122842,0.112962,0.110158,0.112555,0.119003,0.123335,0.118359,0.112981,0.110323,0.113035,0.116900,0.122069,0.116979,0.110103,0.112253,0.109631,0.117962,0.113332,0.110682,0.113077,0.117339,0.121242,0.116757,0.110699,0.112893,0.110123,0.118479,0.110605,0.112842,0.110195,0.116686,0.110014,0.111099,0.113268,0.110620,0.117084,0.110674,0.110545,0.111088
# correct_ssim: 0.745785,0.734357,0.795505,0.744722,0.775172,0.687916,0.757341,0.800618,0.754828,0.780067,0.753339,0.796590,0.756527,0.782316,0.740925,0.799524,0.810201,0.798104,0.779120,0.754564,0.777665,0.799080,0.764440,0.785558,0.761826,0.799740,0.810382,0.800646,0.778696,0.760366,0.781649,0.798450,0.808728,0.797567,0.783972,0.762309,0.783808,0.809099,0.800326,0.810315,0.781443,0.798387,0.808647,0.799076,0.783769,0.767629,0.786145,0.808493,0.799630,0.810078,0.780802,0.807762,0.798740,0.808674,0.785145,0.809151,0.807002,0.798311,0.808327,0.784925,0.808193,0.807683,0.806674
