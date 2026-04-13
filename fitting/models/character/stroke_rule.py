from ..rule import ModelRule
import numpy as np
import math
from .pybpl.objects.part import vanilla_to_motor_with_rotation
import torch


class StrokeTrait:
    def __init__(self):
        self.control = None
        self.scale = None
        self.position = None
        self.angle = None
        self.action = None

    def print(self, out_file=None):
        print('  control = %s\n  scale = %.4f\n  translation = %s\n  angle = %.4f' %
              (np.array_repr(self.control).replace('\n ', ''), self.scale,
               np.array_repr(self.position).replace('\n ', ''), self.angle), file=out_file)
        if out_file is not None:
            out_file.flush()


def normalize(old, new_lb, new_ub):
    old_lb = -1.0
    old_ub = 1.0
    normalized = (new_ub - new_lb) * (old - old_lb) / (old_ub - old_lb) + new_lb
    return normalized


def create_example_traits(token_file):
    import scipy.io as sio
    # import os
    # file_path = os.path.dirname(os.path.abspath(__file__))
    # matfile = '../data/character_token.mat'
    # matfile = '../data/character_token2.mat'
    # matfile = '../data/character_token_4_35.mat'
    # matfile = '../data/character_token_run1_test5.mat'

    # matfile = file_path + '/../../datasets/character/run9_train10_3.mat'

    # matfile = file_path + '/../../datasets/character/data_ncpt_5.mat'

    token = sio.loadmat(token_file, squeeze_me=True)
    positions = token['positions']
    invscales = token['invscales']
    controls = token['shapes']
    motor = token['motor']
    # states = {}
    traits = []
    # stroke = 0
    for i in range(invscales.size):
        if isinstance(invscales[i], float):
            trait = StrokeTrait()
            trait.scale = invscales[i]
            trait.angle = 0.0
            trait.position = positions[i]
            trait.control = controls[i]
            trait.control = trait.control.flatten()
            # states[stroke] = coeff
            traits.append(trait)
            # stroke += 1
        else:
            sub_controls = controls[i]
            sub_motor = motor[i]
            for j in range(invscales[i].size):
                trait = StrokeTrait()
                trait.scale = invscales[i][j]
                trait.angle = 0.0
                trait.control = sub_controls[:, :, j]
                trait.control = trait.control.flatten()
                if j == 0:
                    trait.position = positions[i]
                else:
                    previous_motor = sub_motor[j - 1]
                    trait.position = previous_motor[-1, :]
                # states[stroke] = coeff
                traits.append(trait)
                # stroke += 1
    return traits


class StrokeRule(ModelRule):
    name = 'Stroke'
    # example = create_example_traits()

    def __init__(self, estimator, trait, is_last=False):
        ModelRule.__init__(self, estimator)
        assert isinstance(trait, StrokeTrait)
        self.is_last = is_last
        self.trait = trait

    def generate(self):
        control = torch.tensor(self.trait.control, dtype=torch.float)
        control = control.view(5, 2, 1)
        scale = torch.tensor([self.trait.scale], dtype=torch.float)
        translation = torch.tensor(self.trait.position, dtype=torch.float)
        angle = [self.trait.angle]
        motor = vanilla_to_motor_with_rotation(control, scale, translation, angle)
        self.estimator.add_stroke(motor)

    @staticmethod
    def parse(**kwargs):
        action = kwargs['action']
        # previous_trait = kwargs['previous_trait']
        model_index = kwargs['model_index']
        assert action.size == 14
        default = kwargs['example']
        assert model_index < len(default)
        # is_last = True if (model_index == len(default)-1) else False
        pivot = default[model_index]
        trait = StrokeTrait()
        trait.control = pivot.control + action[0:10] * 5.0
        # coeff.scale = params[10]/2.0 + 0.5
        new_lb = 1.0 / 1.1
        new_ub = 1.0 * 1.1
        factor = normalize(action[10], new_lb, new_ub)
        trait.scale = pivot.scale * factor
        np.clip(trait.scale, 0.01, 0.9)
        trait.position = pivot.translation + action[11:13] * 5
        # coeff.translation = params[11:13] * 20.0
        # coeff.angle = params[13]*math.pi/100.0
        lb = -math.pi / 10.0
        ub = math.pi / 10.0
        trait.angle = normalize(action[13], lb, ub)
        trait.action = action
        return trait

    @staticmethod    
    def num_variables():
        return 14    

    # @staticmethod
    # def max_num_siblings():
    #     return len(StrokeRule.example_traits())

    # @staticmethod
    # def generate_synthetic_data(estimator):
    #     # modeler.generate_sibling_models(theta)
    #     img = estimator.image
    #     img = img.numpy()
    #     img = Image.fromarray(np.uint8(img * 255))
    #     img = img.resize((52, 52))
    #     # plt.imshow(img, cmap=plt.get_cmap('gray'))
    #     # plt.show()
    #     img = np.asarray(img)
    #     indexes = np.nonzero(img >= 128)
    #     if indexes[0].size > 0:
    #         cloud = np.vstack((indexes[0], indexes[1]))
    #         z = np.zeros(indexes[0].shape)
    #         data = np.vstack((cloud, z)).transpose()
    #     else:
    #         assert False
    #     # show_point_cloud(data)
    #     return data

    # @staticmethod
    # def example_traits():
    #     return StrokeRule.example

