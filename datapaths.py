import os

mainpath = '/home/ricardo/Desktop'
datapaths = {'SegThor':os.path.join(mainpath,'SegThor'),
            'StrutSeg2019':os.path.join(mainpath,'StrutSeg2019'),
            'AUTOMI':os.path.join(mainpath,'AUTOMI'),
            'nnUnetAUTOMI':os.path.join(mainpath,'nnUnetAUTOMI', 'dataset', 'nnUnet_raw', 'Dataset001_AUTOMI100'),
            'CityOfHope': os.path.join(mainpath, 'CityOfHope')}
storage_path = '/mnt/storage/ricardo'
original_datasets = {'AUTOMI':os.path.join(storage_path,'ExportDec2021'), 'CityOfHope':os.path.join(storage_path,'CityOfHope')}
resultspath= '/mnt/storage/ricardo/results'
