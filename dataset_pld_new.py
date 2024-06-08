import os
import numpy as np
import cv2
from itertools import combinations
import random
import glob
from typing import Any, Callable, Dict, List, Optional, Tuple
import SimpleITK as sitk
from core.dataproc_utils import resize_image_itk, register, get_aug_crops, standard_normalization, standard_normalization_withmask, resampler_affine_transform


class DataBaseCBV(object):
    def __init__(self,
                 root: str,
                 datafolder: str,
                 side_len=(320, 320, 16),
                 center_shift=(0, 0, 0),
                 data_shape=(256, 288, 16),
                 aug_side=(16, 16, 16),
                 aug_stride=(8, 8, 8),
                 run_stage='train',
                 submodalities=('T1WI', 'T2WI', 'T1_C', 'FLAIR', 'ADC', 'CBV', 'ASL'),
                 cycload=True,
                 use_augment=True,
                 modality_random=False,
                 use_realign=False,
                 return_original=False,
                 randomcrop=(0, 1),
                 randomflip=('sk', 'flr', 'fud', 'r90')):
        self.side_len = side_len
        self.root = root
        self.datafolder = datafolder
        self.center_shift = center_shift
        self.data_shape = data_shape
        self.use_augment = use_augment
        self.modality_random = modality_random
        self.run_stage = run_stage
        self.return_original = return_original
        if not self.use_augment:
            self.aug_model = 'center'
        else:
            self.aug_model = 'random'
        self.aug_side = aug_side
        self.aug_stride = aug_stride
        self.cycload = cycload
        self.use_realign = use_realign
        self.randomcrop = randomcrop
        self.randomflip = randomflip
        self.submodalities = submodalities
        self.cls_num = 2
        self.channels = {'T1WI': 1, 'T2WI': 1, 'T2_FLAIR': 1, 'T1_C': 1, 'ADC': 1, 'ASL': 1, 'bm': 1, 'CBV': 1, 'CBF': 1, 'MTT': 1, 'label': 2}
        self.group_map = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1, 'MCI': 1, 'sSMC': 0,
                          'pSMC': 1, 'SMC': 0, 'sCN': 0, 'pCN': 0, 'ppCN': 1, 'Autism': 1, 'Control': 0}
        self.datapool = {}
        self.input_setup()

    def input_setup(self):
        excludes = ['TAN_YU_JU_2\\2',
         'LIU_XI_MIN_3\\3',
         'GAO_YU_MIN_3\\1',
         'XUE_QING_ZHONG_002150234\\1',
         'LIU_SHENG_LIN_002124213\\1',
         'SHAO_GUANG_CAI_3\\1',
         'NIE_LAN_001952873\\1',
         'WANG_CHANG_YUAN_4\\4',
         'YANG_SI_QIN_002404562\\1',
         'QU_LAN_HUA_2\\1',
         'JIA_CHUAN_ZHEN_002383157\\1',
         'SHI_YUE_CHEN_5\\1',
         'YANG_HAN_YUN_2\\2',
         'XU_ZHAO_GE_4\\2',
         'QU_LAN_HUA_2\\2',
         'LIN_MENG_FEI_4\\4',
         'KANG_XIAN_RONG_6\\4',
         'KANG_XIAN_RONG_6\\3',
         'MA_CUN_CAI_2\\1',
         'YIN_JIAN_KUN_8\\6',
         'JIANG_GUANG_QUAN_2\\2',
         'GAO_YU_MIN_3\\2',
         'DING_PAN_MING_3\\3',
         'YIN_JIAN_KUN_8\\8',
         'DING_XIU_ZHI_6\\1',
         'WANG_CHANG_YUAN_4\\2',
         'XU_FANG_DONG_002356463\\1',
         'LIU_ZONG_HU_4\\4',
         'SHI_YUE_CHEN_5\\4',
         'JI_GUANG_XIAO_4\\2',
         'WU_JIN_SHENG_2\\2',
         'LIU_WEN_MIN_5\\2',
         'WU_LIAN_ZHONG_000445415\\1',
         'GENG_CHUAN_XING_4\\1',
         'JIAO_FANG_AN_3\\2',
         'LIN_YONG_CHENG_002283615\\1',
         'ZHUANG_HONG_JUN_002131788\\1',
         'LIU_GUI_LING_002369907\\1',
         'ZHEN_HONG_BO_002084222\\1',
         'SHAO_GUANG_CAI_3\\3',
         'SHAN_WEN_DOU_3\\3',
         'CHEN_ZHONG_LIAN_2\\1',
         'GENG_HAI_XIA_4\\3',
         'LIN_MENG_FEI_4\\2',
         'CAO_SHU_ZE_7\\6',
         'YAN_HAI_YAN_2\\2',
         'XU_LONG_CHEN_002411797\\1',
         'MENG_JIAN_GUO_001998489\\1',
         'HU_XI_AN_2\\1',
         'WANG_RONG_SHAN_4\\4',
         'LIU_ZONG_HU_4\\3',
         'SONG_HONG_TAO_002324830\\1',
         'DU_CHUN_QING_001364957\\1',
         'ZHANG_CONG_JIN_1\\1',
         'SHAN_WEN_DOU_3\\1',
         'GENG_CHUAN_XING_4\\3',
         'HUANG_DENG_YONG_4\\3',
         'XU_JUN_ZHANG_002276698\\1',
         'XU_YAN_ZHEN_3\\3',
         'JIA_SHU_GUANG_3\\2',
         'FENG_DONG_PING_002422732\\1',
         'LI_YUAN_LIANG_002151683\\1',
         'HAN_SHU_XIANG_002234131\\1',
         'YAO_YU_LIAN_4\\4',
         'LIU_ZHEN_QI_4\\4',
         'GENG_JIAN_GUANG_3\\3',
         'SHI_YUE_CHEN_5\\5',
         'WANG_FENG_XIANG_2\\2',
         'LIANG_JUE_MIN_3\\1',
         'GAO_CHANG_YI_3\\1',
         'CAO_SHU_ZE_7\\1',
         'WANG_YAN_SHI_3\\2',
         'GAO_XI_BIN_1\\1',
         'LIU_ZONG_HU_4\\1',
         'TAN_YU_JU_2\\1',
         'YANG_MEI_FENG_3\\1',
         'LV_FA_FU_002045303\\1',
         'ZHANG_AN_YU_000383267\\1',
         'QIU_MING_YI_2\\2',
         'ZHANG_HAN_XIANG_2\\1',
         'SHAN_WEN_DOU_3\\2',
         'WANG_CHUN_FENG_002440804\\1',
         'GENG_HAI_XIA_4\\4',
         'WANG_XUE_SHEN_002203356\\1',
         'WANG_GUI_XIA_002083978\\1',
         'QU_XIN_HUA_6\\6',
         'CHAO_GUI_YUN_002200602\\1',
         'AI_GUANG_CHEN_002111953\\1',
         'GENG_CHUAN_XING_4\\4',
         'MA_ZUO_LONG_002343068\\1',
         'HUANG_DENG_YONG_4\\2',
         'DUAN_ZHONG_HUA_2\\2',
         'WU_FENG_E_002474397\\1',
         'XU_GEN_SHE_3\\3',
         'SUN_MIN_XIU_6\\2',
         'XIE_SONG_MEI_2\\2',
         'GAO_SHUN_FENG_3\\2',
         'WAN_FU_LI_002111964\\1',
         'JIA_SHU_GUANG_3\\3',
         'JIAO_FANG_AN_3\\3',
         'MU_YU_QING_002368691\\1',
         'TAN_JIN_002098156\\1',
         'WEI_XIU_YING_8\\2',
         'MA_CUN_CAI_2\\2',
         'GUO_XIU_REN_2\\1',
         'QU_XIN_HUA_6\\5',
         'CHEN_LI_HONG_4\\2',
         'CHEN_CUI_LAN_002178995\\1',
         'WANG_HUAN_CHEN_5\\4']

        if self.run_stage == 'train':
            imdb_dsc1 = [[[os.path.join(groupdir, subdir), 0] for subdir in os.listdir(os.path.join(self.processed_folder, groupdir)) if os.path.isdir(os.path.join(self.processed_folder, groupdir, subdir))] for groupdir in os.listdir(self.processed_folder)]
            # self.imdb = [it for db in imdb_dsc1 for it in db if it[0] not in excludes]
            self.imdb = [it for db in imdb_dsc1 for it in db]

        elif self.run_stage == 'test':
            imdb_dsc1 = [[[os.path.join(groupdir, subdir), 0] for subdir in os.listdir(os.path.join(self.processed_folder, groupdir)) if os.path.isdir(os.path.join(self.processed_folder, groupdir, subdir))] for groupdir in os.listdir(self.processed_folder)]
            self.imdb = [it for db in imdb_dsc1 for it in db]
        else:
            imdb_dsc1 = [[[os.path.join(groupdir, subdir), 0] for subdir in os.listdir(os.path.join(self.processed_folder, groupdir)) if os.path.isdir(os.path.join(self.processed_folder, groupdir, subdir))] for groupdir in os.listdir(self.processed_folder)]
            self.imdb = [it for db in imdb_dsc1 for it in db]
        print(len(self.imdb))

    def input_setup_1(self):
        # imdb_dsc0 = [[os.path.join('for_figures', subdir, datedir), 0] for subdir in os.listdir(os.path.join(self.input_path, 'for_figures'))
        #              for datedir in os.listdir(os.path.join(self.input_path, 'for_figures', subdir))]
        # imdb_dsc1 = [[os.path.join('Single_PLD_ASL', subdir), 0] for subdir in os.listdir(os.path.join(self.processed_folder, 'Single_PLD_ASL'))]
        # imdb_dsc2 = [[os.path.join('Multi_PLD_ASL', subdir), 1] for subdir in os.listdir(os.path.join(self.processed_folder, 'Multi_PLD_ASL'))]
        imdb_dsc1 = [[[os.path.join(groupdir, subdir), 0] for subdir in os.listdir(os.path.join(self.processed_folder, groupdir)) if os.path.isdir(os.path.join(self.processed_folder, groupdir, subdir))] for groupdir in os.listdir(self.processed_folder)]
        # print(imdb_dsc2)
        # self.imdb_train = imdb_dsc1[0:len(imdb_dsc1):3] + imdb_dsc1[1:len(imdb_dsc1):3]
        # self.imdb_valid = imdb_dsc1[len(imdb_dsc1)//3*2::]
        # self.imdb_test = imdb_dsc1[len(imdb_dsc1)//3*2::]
        if self.run_stage == 'train':
            # self.imdb = imdb_dsc1[0:len(imdb_dsc1):3] + imdb_dsc1[1:len(imdb_dsc1):3]
            self.imdb = [it for db in imdb_dsc1 for it in db[0:len(db)//4*3]]
        elif self.run_stage == 'test':
            # self.imdb = imdb_dsc1[2:len(imdb_dsc1):3]
            self.imdb = [it for db in imdb_dsc1 for it in db[len(db)//4*3::]]
        else:
            self.imdb = imdb_dsc1
        print(len(self.imdb))

    # def seg_by_freesurfer(self, path, T1WI):
    #     # ---------------------------------
    #     mri_convert T1WI.nii.gz orig.nii.gz --conform
    #     mri_add_xform_to_header -c talairach.xfm orig.nii.gz orig.nii.gz
    #     mri_nu_correct.mni --no-rescale --i orig.nii.gz --o orig_nu.nii.gz --ants-n4 --n 1 --proto-iters 1000 --distance 50
    #     talairach_avi --i orig_nu.nii.gz --xfm talairach.auto.xfm --atlas 3T18yoSchwartzReactN32_as_orig
    #     cp talairach.auto.xfm talairach.xfm
    #     lta_convert --src orig.nii.gz --trg  $FREESURFER_HOME//average/mni305.cor.mgz --inxfm talairach.xfm --outlta talairach.xfm.lta --subject fsaverage --ltavox2vox
    #
    #     talairach_afd -T 0.005 -xfm talairach.xfm
    #     awk -f $FREESURFER_HOME//bin/extract_talairach_avi_QA.awk talairach_avi.log
    #     tal_QC_AZS talairach_avi.log
    #
    #     mri_nu_correct.mni --i orig.nii.gz --o nu.nii.gz --uchar talairach.xfm --proto-iters 1000 --distance 50 --n 1 --ants-n4
    #     mri_add_xform_to_header -c talairach.xfm nu.nii.gz nu.nii.gz
    #
    #     mri_normalize -g 1 -seed 1234 -mprage nu.nii.gz T1.nii.gz
    #     mri_em_register -skull nu.nii.gz  $FREESURFER_HOME//average/RB_all_withskull_2020_01_02.gca talairach_with_skull.lta
    #     mri_watershed -T1 -brain_atlas $FREESURFER_HOME//average/RB_all_withskull_2020_01_02.gca talairach_with_skull.lta T1.nii.gz brainmask.nii.gz

    def subject_preprocessing(self, flnm):
        source_images = {'T1WI': ['T1WI',], 'T2WI': ['T2WI'], 'FLAIR': ['T2_FLAIR', 'T2WI_FLAIR'],
                         'T1_C': ['T1WI_C', 'T1WI+C', 'T1_C', ], 'ADC': ['ADC', ],
                         'CBV': ['CBV', ], 'ASL': ['ASL', 'TGSE'], 'CBF': ['CBF',], 'MTT': ['MTT']}

        align_images = {'T1WI': ['rT1WI', 'T1WI'], 'T2WI': ['rT2WI', 'T2WI'], 'FLAIR': ['rT2_FLAIR', 'T2_FLAIR', 'T2WI_FLAIR'],
                        'T1_C': ['rT1_C', 'T1WI_C', 'T1WI+C', 'T1_C', ], 'ADC': ['rADC', 'ADC'], 'ASL': ['rASL', 'rTGSE', 'ASL', 'TGSE'],
                        'CBV': ['rCBV', 'CBV', ], 'CBF': ['rCBF', 'CBF'], 'MTT': ['rMTT', 'MTT']}

        if self.use_realign:
            dataessamble = align_images
        else:
            dataessamble = source_images
        T1path = os.path.join(self.processed_folder, flnm, 'T1WI.nii.gz')
        spacing = [0.6875, 0.6875, None]
        # spacing = [0.3594, 0.3594, None]
        newsize = [320, 320, None]
        T1image = resize_image_itk(sitk.ReadImage(T1path), newSpacing=spacing, newSize=newsize)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(T1image)
        bmpath = os.path.join(self.processed_folder, flnm, 'brainmask.nii.gz')
        if not os.path.exists(bmpath):
            bmpath = os.path.join(self.processed_folder, flnm, 'bmT1WI.nii.gz')
        # brainmask = sitk.ReadImage(bmpath)
        brainmask = resampler.Execute(sitk.ReadImage(bmpath))
        sitk.WriteImage(brainmask > 0, os.path.join(self.processed_folder, flnm, 'rBM' + '.nii.gz'))
        for ids in dataessamble:
            dataessamble[ids] = dataessamble[ids] if isinstance(dataessamble[ids], list) else [dataessamble[ids],]
            moda = [m for m in dataessamble[ids] if os.path.exists(os.path.join(self.processed_folder, flnm, m + '.nii.gz'))]
            if len(moda) == 0: continue
            fullpath = os.path.join(self.processed_folder, flnm, moda[0] + '.nii.gz')
            fullpathr = os.path.join(self.processed_folder, flnm, ids + 'n.nii.gz')

            orig_image = sitk.ReadImage(fullpath)
            if len(orig_image.GetSize()) == 4:
                orig_image = sitk.Maximum(orig_image[:,:,:,0], 0)
            else:
                orig_image = sitk.Maximum(orig_image, 0)
            # print(flnm, dataessamble[ids], orig_image.GetSpacing(), )
            # input_image = resize_image_itk(orig_image, newSpacing=spacing, newSize=newsize)

            rsz_image = resampler.Execute(orig_image)
            rsz_image = sitk.Cast(rsz_image, sitk.sitkFloat32) #* sitk.Cast(brainmask, sitk.sitkFloat32)
            # std_image = standard_normalization(rsz_image, remove_tail=False, divide='mean')
            std_image = standard_normalization_withmask(sitk.Cast(rsz_image, sitk.sitkFloat32), brainmask, remove_tail=False, divide='mean')
            std_image = sitk.Cast(std_image*1000, sitk.sitkInt16)
            sitk.WriteImage(std_image, fullpathr)

    def read_images(self, flnm):
        source_images = {'T1WI': ['T1WI', 'T1WI'], 'T2WI': 'T2WI', 'FLAIR': ['T2_FLAIR', 'T2WI_FLAIR'], 'T1_C': ['T1WI_C', 'T1WI+C', 'T1_C'], 'ADC': 'ADC', 'CBV': ['CBV', ], 'ASL': ['ASL', 'TGSE', ], 'brainmask': ['bmT1WI', 'brainmask']}
        target_images = {'CBF': ['CBF',], 'MTT': ['MTT',]}
        dataessamble = dict(source_images, ** target_images)
        T1path = os.path.join(self.processed_folder, flnm, 'T1WIn.nii.gz')

        if not os.path.exists(T1path):
            self.subject_preprocessing(flnm)
        # print(flnm)
        self.subject_preprocessing(flnm)
        # bmpath = os.path.join(self.input_path, flnm, 'bmT1WI.nii.gz')

        # brainmask = resize_image_itk(sitk.ReadImage(bmpath), newSpacing=spacing, newSize=newsize)
        viewsample = []
        for ids in dataessamble:
            dataessamble[ids] = dataessamble[ids] if isinstance(dataessamble[ids], list) else [dataessamble[ids], ]
            moda = [m for m in dataessamble[ids] if os.path.exists(os.path.join(self.processed_folder, flnm, m + '.nii.gz')) or os.path.exists(os.path.join(self.processed_folder, flnm, 'r' + m + '.nii.gz'))]
            if len(moda) == 0:
                dataessamble[ids] = None
                continue
            fullpath = os.path.join(self.processed_folder, flnm, moda[0] + '.nii.gz')
            if ids in self.submodalities:
                fullpathr = os.path.join(self.processed_folder, flnm, ids + 'n.nii.gz')
                if os.path.exists(fullpathr):
                    std_image = sitk.ReadImage(fullpathr)
                    input_data = np.transpose(np.float32(sitk.GetArrayFromImage(std_image))/1000.0)
                    viewsample.append(input_data[:, :, np.shape(input_data)[2] // 2])
                    dataessamble[ids] = std_image
                else:
                    dataessamble[ids] = None
            else:
                if ids in ['brainmask', ]:
                    fullpath = os.path.join(self.processed_folder, flnm, 'rBM' + '.nii.gz')
                    std_image = sitk.ReadImage(fullpath)
                    input_data = np.transpose(np.float32(sitk.GetArrayFromImage(std_image)))
                    viewsample.append(input_data[:, :, np.shape(input_data)[2] // 2])
                    dataessamble[ids] = std_image

        viewpath = os.path.join(self.processed_folder, flnm.replace('\\', '_').replace('/', '_') + '.png')
        # cv2.imwrite(viewpath, np.concatenate(viewsample, axis=0)*255)

        return dataessamble

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        flnm, group = self.imdb[index][0:2]
        # print(flnm)
        if flnm in self.datapool:
            dataessamble = self.datapool[flnm]
        else:
            dataessamble = self.read_images(flnm)
            label = np.zeros(2, np.float32)
            label[group] = 1
            dataessamble.update({'label': label})
            if self.cycload:
                self.datapool[flnm] = dataessamble
        refdata = dataessamble[self.submodalities[0]]

        aug_side = self.aug_side
        aug_step = np.maximum(self.aug_stride, 1)
        image_size = refdata.GetSize()
        aug_range = [min(aug_side[dim], (image_size[dim] - self.data_shape[dim] - self.center_shift[dim]) // 2) for dim in range(3)]
        aug_center = [(image_size[dim] + self.center_shift[dim] - self.data_shape[dim]) // 2 for dim in range(3)]

        aug_crops, count_of_augs = get_aug_crops(aug_center, aug_range, aug_step, aug_index=(1,), aug_model=self.aug_model)
        datainput = {}


        theta, shear, scale = [0] * 3, [0] * 3, [1] * 3
        if self.use_augment:
            theta, shear = [random.randint(-5, 5) for _ in [1] * 3], [0] * 3#[random.random() * 0.1 - 0.05 for _ in [1] * 3]
            scale = [random.choice([-1, 1]) * (random.random() * 0.4 + 0.8) for _ in [1] * 3]

        resampler = resampler_affine_transform(refdata, self.data_shape, origin=aug_crops[0], theta=theta, shear=shear, scale=scale)
        for item in dataessamble:
            if self.modality_random:
                theta_shift, shear = [random.randint(-5, 5)+v for v in theta], [0] * 3  # [random.random() * 0.1 - 0.05 for _ in [1] * 3]
                origin_shift = [(random.random()*0.2-0.1) * image_size[idx]+aug_crops[0][idx] for idx in range(len(image_size))]
                resampler = resampler_affine_transform(refdata, self.data_shape, origin=origin_shift, theta=theta_shift, shear=shear, scale=scale)
            if isinstance(dataessamble[item], sitk.Image):
                Interpolator = sitk.sitkNearestNeighbor if item in ['mask', 'brainmask', ] else sitk.sitkLinear
                resampler.SetInterpolator(Interpolator)
                resampler.SetOutputPixelType(dataessamble[item].GetPixelID())
                if self.return_original:
                    datainput[item] = dataessamble[item]
                else:
                    datainput[item] = resampler.Execute(dataessamble[item])

        for item in datainput:
            datainput[item] = np.float32(sitk.GetArrayFromImage(datainput[item])[np.newaxis])
            if item not in ['mask', 'brainmask', 'seg']:
                datainput[item] = datainput[item]*1.0/2000.0-0.0

        # end_time = time.time()
        # print('time cost: %.4f seconds.' % (end_time - start_time))

        return datainput, flnm

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.datafolder, )

    @property
    def processed_folder(self) -> str:
        folder = os.path.join(self.root, self.datafolder, )
        if not os.path.exists(folder):
            os.mkdir(folder)
        return folder

    def __len__(self) -> int:
        return len(self.imdb)

    def save_output(self, result_path, flnm, eval_out):
        flnm, refA, refB, synA, synB = eval_out['flnm'], eval_out['refA'], eval_out['refB'], eval_out['synA'], eval_out['synB']
        T1path = os.path.join(self.input_path, flnm, 'T1WIr.nii.gz')
        T1image = sitk.ReadImage(T1path)
        if isinstance(flnm, bytes): flnm = flnm.decode()
        result_path = result_path.replace('dsc_new', 'for_figures')
        if not os.path.exists(result_path + "/{0}".format(flnm)): os.makedirs(result_path + "/{0}".format(flnm))
        synA = (synA + 1.0) * 128.0
        synB = (synB + 1.0) * 128.0
        refA = (refA + 1.0) * 128.0
        refB = (refB + 1.0) * 128.0
        for ref in ['refA', 'refB', 'synA', 'synB']:
            img = eval_out[ref]
            if img is not None:
                img = sitk.GetImageFromArray(np.round((img[0]+1.0) * 1000))
                # img = sitk.Cast(img, sitk.sitkInt16)
                # img = sitk.GetImageFromArray(self.ct_rgb2gray(img))
                img.SetOrigin(T1image.GetOrigin())
                img.SetSpacing(T1image.GetSpacing())
                img.SetDirection(T1image.GetDirection())
                sitk.WriteImage(img, result_path + "/{0}/{1}.nii.gz".format(flnm, ref), useCompression=True)
        # if eval_out['synA'] is not None and eval_out['synB'] is not None and eval_out['refA'] is not None:
            # cv2.imwrite(result_path + "/{0}/CBV_sample.png".format(flnm),
            #             np.concatenate((refA[0, 13, :, :], synA[0, 13, :, :], synB[0, 13, :, :]), axis=0))

# ('BI_SI_TIAN_002563009\\UNKNOWN',) CBV loss: 0.218711,0.213589,0.207678,0.216801,0.214497,0.015571,0.212968,0.207968,0.215635,0.213257,0.218711,0.206036,0.211839,0.210901,0.213589,0.207161,0.204878,0.207678,0.213245,0.216801,0.214497,0.206911,0.212286,0.210630,0.212968,0.207836,0.205768,0.207968,0.213036,0.215635,0.213257,0.206209,0.204685,0.206036,0.210568,0.211839,0.210901,0.205338,0.207161,0.204878,0.213245,0.207019,0.205510,0.206911,0.210606,0.212286,0.210630,0.206089,0.207836,0.205768,0.213036,0.205096,0.206209,0.204685,0.210568,0.205338,0.205875,0.207019,0.205510,0.210606,0.206089,0.205096,0.205875 psnr: 18.550926,18.783716,19.088393,18.602021,18.741313,47.243205,18.797737,19.090848,18.671096,18.789663,18.550926,19.164649,18.839842,18.896866,18.783716,19.116446,19.257018,19.088393,18.768240,18.602021,18.741313,19.127863,18.823841,18.907402,18.797737,19.095062,19.204030,19.090848,18.769762,18.671096,18.789663,19.159034,19.239397,19.164649,18.901941,18.839842,18.896866,19.213893,19.116446,19.257018,18.768240,19.126288,19.194955,19.127863,18.897831,18.823841,18.907402,19.174773,19.095062,19.204030,18.769762,19.210984,19.159034,19.239397,18.901941,19.213893,19.172046,19.126288,19.194955,18.897831,19.174773,19.210984,19.172046 rmae: 0.118155,0.115031,0.111066,0.117462,0.115594,0.004343,0.114845,0.111034,0.116532,0.114952,0.118155,0.110095,0.114290,0.113542,0.115031,0.110708,0.108930,0.111066,0.115236,0.117462,0.115594,0.110562,0.114501,0.113404,0.114845,0.110981,0.109597,0.111034,0.115216,0.116532,0.114952,0.110166,0.109152,0.110095,0.113476,0.114290,0.113542,0.109473,0.110708,0.108930,0.115236,0.110582,0.109712,0.110562,0.113529,0.114501,0.113404,0.109967,0.110981,0.109597,0.115216,0.109509,0.110166,0.109152,0.113476,0.109473,0.110001,0.110582,0.109712,0.113529,0.109967,0.109509,0.110001 ssim: 0.758810,0.768195,0.780915,0.755876,0.763936,0.999700,0.773241,0.783996,0.763047,0.771041,0.758810,0.786353,0.773541,0.776327,0.768195,0.784941,0.790773,0.780915,0.768375,0.755876,0.763936,0.786083,0.773624,0.778870,0.773241,0.784843,0.789750,0.783996,0.771448,0.763047,0.771041,0.787741,0.791162,0.786353,0.778039,0.773541,0.776327,0.790600,0.784941,0.790773,0.768375,0.786697,0.790189,0.786083,0.779103,0.773624,0.778870,0.789537,0.784843,0.789750,0.771448,0.791188,0.787741,0.791162,0.778039,0.790600,0.790101,0.786697,0.790189,0.779103,0.789537,0.791188,0.790101
