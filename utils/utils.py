import pandas as pd

def make_tif_list(xls_file,animal_names):
    '''list all the tif files present in the xls info filefor each of the given animals'''

    data = pd.read_excel(xls_file)
    root_folder = '/ceph/imaging1/arie'
    tif_file_list = []

    for animal in animal_names:
        subset = data[data['name']==animal]
        print (f'{len(subset)} files found for mouse: {animal}')
        for i in range(len(subset)):
            animal_num = subset['mouse'].iloc[i]
            year = subset['year'].iloc[i]
            month = str(subset['month'].iloc[i]).zfill(2)
            day = str(subset['day'].iloc[i]).zfill(2)
            rec = subset['rec'].iloc[i]
            tif_file = subset['filename.tif'].iloc[i]
            session = f'{year}{month}{day}'
            tif_path = root_folder+f'/{animal_num}_{animal}/{session}_{animal_num}/{tif_file}.tif'
            tif_file_list.append(tif_path)
    
    return tif_file_list

