from ..rule import ModelRule
import numpy as np
from .pybpl.objects.part import vanilla_to_motor_with_rotation
import torch
from PIL import Image
from easydict import EasyDict
from tools.tool import rescale
from .pybpl.library import Library
from .pybpl.model.image_dist import CharacterImageDist
from tools.geometry import get_boundary, show_point_cloud, get_2D_rotation_matrix


def generate_stroke(nsub, control, scale, position, angle):
        assert nsub == scale.size
        control = torch.tensor(control, dtype=torch.float)
        scale = torch.tensor(scale, dtype=torch.float)
        motor = vanilla_to_motor_with_rotation(control, scale, position, angle)
        return motor


class Stroke:
    def __init__(self):
        self.num_sub_strokes = None
        self.control = None
        self.scale = None
        self.position = None
        self.angle = None
        self.relation = None

    def print(self, out_file=None):
        print('  control = %s\n  scale = %.4f\n  translation = %s\n  angle = %.4f' %
              (np.array_repr(self.control).replace('\n ', ''), self.scale,
               np.array_repr(self.position).replace('\n ', ''), self.angle), file=out_file)
        if out_file is not None:
            out_file.flush()

class CharacterTrait:
    def __init__(self):
        self.ink_ncon = 2  # image broadening parameters: number of convolutions
        self.angle = 0.  # global rotation, np.pi/2
        self.scales = np.ones(2)
        self.strokes = []
        self.action = None

    def print(self, out_file=None):
        print('  control = %s\n  scale = %.4f\n  translation = %s\n  angle = %.4f' %
              (np.array_repr(self.control).replace('\n ', ''), self.scale,
               np.array_repr(self.translation).replace('\n ', ''), self.angle), file=out_file)
        if out_file is not None:
            out_file.flush()


class CharacterRule(ModelRule):

    def __init__(self, estimator=None, name=None):
        ModelRule.__init__(self, estimator)
        self.trait = None
        self.pivot_trait = None
        self.name = name
        self.num_variables = 0
        # self.stroke_action_size = StrokeRule.num_variables()
        self.num_strokes = None
        self.image_dist = CharacterImageDist(Library(use_hist=True))
        self.motors = []
        self.load_pivot()

    def generate(self):
        assert self.motors
        self.image_dist.ps.ink_ncon = self.trait.ink_ncon
        img = self.image_dist.get_image(self.motors, self.trait.scales)
        img = img.numpy()
        img = Image.fromarray(np.uint8(img * 255))
        img = img.rotate(self.trait.angle/np.pi*180)
        img_array = np.asarray(img)
        indexes = np.nonzero(img_array > 127)
        # center = np.array([26, 26])  # 26 ~= 105/4
        if indexes[0].size > 0:
            cloud = np.vstack((indexes[0], indexes[1]))
            cloud = cloud/2.0 + 0.25
            cloud = cloud.transpose()
            # center = np.array([26, 26])  # 26 ~= 105/4
            # center = np.array([26.25, 26.25])  # 26.25=105/4
            # rotation = get_2D_rotation_matrix(self.state.angle, center)
            # rotation = np.asarray(rotation)
            # aug = np.ones(cloud.shape[1])
            # # aug = np.ones(cloud.shape[0])
            # cloud_aug = np.vstack((cloud, aug))
            # # cloud_aug = np.vstack((cloud.transpose(), aug))
            # cloud = (rotation @ cloud_aug).transpose()
            # cloud = cloud - center + self.trait.position
            # cloud = cloud + self.trait.position        
        else:
            cloud = np.array([-1.0, -1.0])
            cloud = cloud.reshape(1, -1)
            # assert False

        # show_point_cloud(cloud)
        # img = img.resize((int(img.size[0] / 2), int(img.size[1] / 2)))
        # position = self.trait.position - center        
        # self.estimator.model_image.paste(img, (int(position[0]), int(position[1])))        
        self.estimator.add_model(new_model=cloud)

    def parse(self, action):
        cfg = self.estimator.cfg['rule']
        assert action.size == self.num_variables
        trait = CharacterTrait()
        count = 0
        trait.ink_ncon = int(rescale(action[count], 0, cfg['max_ink']))  # ink thickness
        count += 1
        trait.angle = rescale(action[count], 0, cfg['max_global_rotation'])  # global rotation
        count += 1
        # global scale_x, global scale_y
        trait.scales = rescale(action[count:count+2], 0., cfg['max_global_scale'])
        count += 2

        self.motors = []
        global_translation = np.zeros(2)
        for i in range(self.num_strokes):
            pivot = self.pivot_trait.strokes[i]
            if isinstance(pivot.relation, str):  # independent stroke
                assert pivot.relation == 'independent' 

                if i == 0:  # first stroke
                    # image_size = self.estimator.data_image.shape
                    # image_size = np.array(image_size, dtype=np.float32)
                    # position = rescale(action[count:count+2], -1, image_size+1)   # position
                    # global_translation = position - pivot.position
                    global_translation = cfg['max_global_translation'] * action[count:count+2]
                    position = pivot.position + global_translation
                else:
                    local_translation = cfg['max_local_translation'] * action[count:count+2]
                    position = pivot.position + local_translation + global_translation

                position = torch.tensor(position, dtype=torch.float)
                count += 2
            elif pivot.relation[0] == 'mid':
                assert len(self.motors) >= 1
                attach_spot = pivot.relation[1] - 1
                subid_spot = pivot.relation[2] - 1
                neval = self.motors[attach_spot].shape[1]                
                eval_spot_shift = rescale(action[count], -cfg['max_eval_spot'], cfg['max_eval_spot'])
                eval_spot = np.clip(pivot.relation[-1] + eval_spot_shift, 2., 6.)
                # TODO
                spot = int(rescale(eval_spot, 0, neval-1, old_lb=2, old_ub=6))
                position = self.motors[attach_spot][subid_spot, spot]                
                count += 1
            elif pivot.relation[0] == 'start':
                assert len(self.motors) >= 1
                attach_spot = pivot.relation[1] - 1
                position = self.motors[attach_spot][0, 0, :]
            elif pivot.relation[0] == 'end':
                assert len(self.motors) >= 1
                attach_spot = pivot.relation[1] - 1
                position = self.motors[attach_spot][-1, -1, :]                
            else:
                assert False                
            nsub = pivot.num_sub_strokes
            angle = rescale(action[count:count+1*nsub], -cfg['max_local_rotation'], cfg['max_local_rotation'])  # stroke rotation
            count += 1*nsub               
            scale = rescale(action[count:count+1*nsub], 1/cfg['max_local_scale'], cfg['max_local_scale'])  # stroke scale
            count += 1*nsub
            scale = np.clip(pivot.scale * scale, 0.01, 10.)  # TODO: check the scale                         
            control = rescale(action[count:count+10*nsub], -1., 1.) * cfg['max_control']  # control
            control = pivot.control + np.reshape(control, pivot.control.shape)
            count += 10*nsub
            motor = generate_stroke(nsub, control, scale, position, angle)
            self.motors.append(motor)                 
        assert count == action.size
        self.trait = trait
        # self.trait = self.pivot_trait
        return trait

    def get_num_variables(self):
        return self.num_variables

    @staticmethod
    def generate_synthetic_data(modeler):
        # modeler.generate_sibling_models(theta)
        img = modeler.estimator.image
        img = img.numpy()
        img = Image.fromarray(np.uint8(img * 255))
        img = img.resize((52, 52))
        # plt.imshow(img, cmap=plt.get_cmap('gray'))
        # plt.show()
        img = np.asarray(img)
        indexes = np.nonzero(img >= 128)
        if indexes[0].size > 0:
            cloud = np.vstack((indexes[0], indexes[1]))
            z = np.zeros(indexes[0].shape)
            data = np.vstack((cloud, z)).transpose()
        else:
            assert False
        # show_point_cloud(data)
        return data

    def load_pivot(self):
        import scipy.io as sio
        token_file = self.estimator.cfg['rule']['token_file']
        token = sio.loadmat(token_file, squeeze_me=True)
        positions = token['positions_token']
        invscales = token['invscales_token']
        controls = token['shapes_token']
        relation = token['relation']
        
        character_trait = CharacterTrait()
        strokes = []

        # character-wise variables: ink, 1; global rotation, 1; global scale, 2;
        self.num_variables = 4  

        # why invscales.size-1? see the explanation in the Matlab code: datasets/character/get_character_trait.m
        self.num_strokes = invscales.size - 1        
        for i in range(self.num_strokes):
            stroke = Stroke()
            stroke.relation = relation[i]
            stroke.position = positions[i]                                        
            if isinstance(stroke.relation, str):  # independent stroke
                assert stroke.relation == 'independent'                
                self.num_variables += 2  # stroke-wise variables: positions
            elif stroke.relation[0] == 'mid':
                self.num_variables += 1  # stroke-wise variable: eval_spot_token
                assert stroke.relation[-1] >= 2. and stroke.relation[-1] <= 6.
            else:
                assert stroke.relation[0] == 'start' or stroke.relation[0] == 'end'                             
            if isinstance(invscales[i], float):  # the stroke has only one sub-stroke
                stroke.num_sub_strokes = 1                
                # stroke.angle = [0.]
                stroke.scale = np.array((invscales[i],))
                stroke.control = np.expand_dims(controls[i], axis=-1)                                 
            else:
                stroke.num_sub_strokes = len(invscales[i])
                stroke.scale = invscales[i]
                # stroke.angle = np.zeros(stroke.scale.shape)
                stroke.control = controls[i]                                  
            self.num_variables += 12 * stroke.num_sub_strokes # stroke-wise variable: angle, 1; scale, 1; shape, 10
            strokes.append(stroke)

        character_trait.strokes = strokes
        self.pivot_trait = character_trait


