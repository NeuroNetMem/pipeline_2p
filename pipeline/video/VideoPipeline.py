import json
import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF

class VideoPipeline(parameters_file):
    
    def __init__(self):
        # reads parameter file
        f = open(parameters_file)
        self.parameters = json.load(f)
        f.close()
        # starts caiman process
        n_processes = psutil.cpu_count()
        self.c, self.dview, self.n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,
                                                 single_thread=False)


    def crop(self):
        '''Crops video, saves video and parameters'''
        crop_parameters = self.parameters['crop']

        pass

    def motion_correct(self,input_video_file):
        '''Runs motion correction, saves video and parameters'''

        print(f'Performing motion correction on {input_video_file}')
        

        # Loads parameters for motion correction
        parameters = self.parameters['motion_correction']
        # Calculate movie minimum to subtract from movie
        min_mov = np.min(cm.load(input_video_file))
        parameters['min_mov'] = min_mov
        opts = params.CNMFParams(params_dict = parameters)

        # Create a MotionCorrect object
        mc = MotionCorrect([input_video_file], dview=self.dview, **opts.get_group('motion'))
        # Perform rigid motion correction
        mc.motion_correct_rigid(save_movie=parameters['save_movie_rig'], template=None)
        # Obtain template, rigid shifts and border pixels
        total_template_rig = mc.total_template_rig
        shifts_rig = mc.shifts_rig
        output['meta']['cropping_points'] = [0,0,0,0]


        return

    def extract_sources():
        '''Runs motion correction, saves cropped video and used cropped parameters'''
        pass



