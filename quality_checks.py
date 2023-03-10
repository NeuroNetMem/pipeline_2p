import caiman as cm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import gc
import utils.utils as ut



xls_info_file = '/ceph/imaging1/arie/fileinfo.xlsx'
animals = ['toms'] #animals to analize
output_path = '/ceph/imaging1/davide/2p_data/data_quality'
saturation_th = - 100
log_file = output_path + '/log.txt'


#tif_files = ut.make_tif_list(xls_info_file,animals)

tif_files = ['/ceph/imaging1/arie/429420_toms/20230203_429420/20230203_429420_00002.tif', 
             '/ceph/imaging1/arie/429420_toms/20230202_429420/20230202_429420_00001.tif',
             '/ceph/imaging1/arie/429419_croc/20230202_429419/20230202_429419_00002.tif',
             '/ceph/imaging1/arie/429419_croc/20230202_429419/20230202_429419_00002.tif'
            ]

Path(output_path).mkdir(parents=True, exist_ok=True)

data_quality = {'animal':[],'session':[],'tif_file':[],'saturation_fraction': [],
                'min_val':[],'max_val':[]}

for i,filepath in enumerate(tif_files):

    #check if file exists, if not save in txt
    if not Path(filepath).is_file():
        print(f'Not found : {filepath},skipping')
        continue


    session_data = filepath.split('/')
    animal = session_data[-3]
    session = session_data[-2].split('_')[0]
    video_name = session_data[-1].split('.')[0]

    save_path = output_path+f'/{animal}/{session}'

    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(f'Processing animal: {animal},session {session}, {i+1}/{len(tif_files)}')

    print('loading movie ...')
    movie = cm.load(filepath)
    print(f'movie shape: {movie.shape}')

    print('computing quality measures ...')
    movie_min = np.min(np.asarray(movie).flatten())
    movie_max = np.max(np.asarray(movie).flatten())
    mean_activity = np.mean(movie,axis=(1,2))
    saturation_frac = np.sum(mean_activity>saturation_th)/len(mean_activity)

    data_quality['animal'].append(animal)
    data_quality['session'].append(session)
    data_quality['tif_file'].append(video_name)
    data_quality['saturation_fraction'].append(saturation_frac)
    data_quality['min_val'].append(movie_min)
    data_quality['max_val'].append(movie_max)

    
    print('making timecourse plot ...')
    
    # AVERAGE ACTIVITY TIMECOURSE
     
    plt.figure(figsize=(15,5))
    plt.plot(mean_activity)
    plt.title(f'saturation fraction: {saturation_frac}')
    plt.xlabel('time (frames)')
    plt.ylabel('average pixel value')
    plt.savefig(save_path+f'/{video_name}_luminance_timecourse.png')
    
    # SUMMARY IMAGES TIMECOURSE
    
    print('computing mean image ...')
    mean_image = np.mean(movie,axis=0)

    
    plt.figure(figsize=(10,10))
    plt.title('Mean image')
    plt.imshow(mean_image,cmap='gray')
    plt.savefig(save_path+f'/{video_name}_mean_img.png')
    
    #print('compute correlation adn pnr images ...')
    #correlation_image,pnr_image = cm.summary_images.correlation_pnr(movie,swap_dim=False)
    
    #print('making summary images plot ...')
    
    #plt.subplot(1,2,1)
    #plt.title('Corr image')
    #plt.imshow(correlation_image,cmap='gray')
    
    #plt.subplot(1,2,2)
    #plt.title('PNR image')
    #plt.imshow(pnr_image,cmap='gray')
    
    #plt.tight_layout()
    
    
    #plt.savefig(save_path+f'/{video_name}_summary_imgs.png')


    # clear memory
    del(movie)
    gc.collect()

print('saving quality measures ...')
data_quality = pd.DataFrame(data_quality)
data_quality.to_csv(output_path+'data_quality.csv')
