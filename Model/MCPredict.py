import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet
from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance
from pymatgen.io.ase import AseAtomsAdaptor

class MCAtomsData(object):
    def __init__(self, atoms, atom_init_path, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(atom_init_path), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_path)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.atoms = atoms
        
        crystal = AseAtomsAdaptor.get_structure(atoms)
        
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea = [], [] # nbr_fea = distance
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        
        self.atom_fea = torch.Tensor(atom_fea)
        self.nbr_fea = torch.Tensor(nbr_fea)
        self.nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        self.crystal_atom_idx=[torch.LongTensor(np.arange(len(atoms)))]
        
    def MakeCGCNNInput(self,atomic_numbers):
        atom_fea = np.vstack([self.ari.get_atom_fea(an) for an in atomic_numbers])
        atom_fea = torch.Tensor(atom_fea)
        return atom_fea, self.nbr_fea, self.nbr_fea_idx, self.crystal_atom_idx


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
    
    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
class Predictor4MC(object):
    def __init__(self,atoms,modelpath,atom_init_path,disable_cuda = False):
        self.atoms = atoms
        self.modelpath = modelpath
        self.cuda = not disable_cuda and torch.cuda.is_available()
        
        # load data
        self.MCdatamaker = MCAtomsData(atoms, atom_init_path)
        
        # optionally resume from a checkpoint
        if os.path.isfile(modelpath):
            print("=> loading model '{}'".format(modelpath))
            checkpoint = torch.load(modelpath,
                                    map_location=lambda storage, loc: storage)
            model_args = argparse.Namespace(**checkpoint['args'])
            print("=> loaded model '{}' (epoch {}, validation {})"
                  .format(modelpath, checkpoint['epoch'],
                          checkpoint['best_mae_error']))
        else:
            print("=> no model found at '{}'".format(modelpath))
            raise ValueError
            
            
        # build model
        structures = (self.MCdatamaker.atom_fea, self.MCdatamaker.nbr_fea, self.MCdatamaker.nbr_fea_idx)
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        self.model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                    'classification' else False)
        self.normalizer = Normalizer(torch.zeros(3))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.normalizer.load_state_dict(checkpoint['normalizer'])
        if self.cuda:
            self.model.cuda()
        
        self.model.eval()
        
    def GetEnergy(self,atomic_numbers):
        input = self.MCdatamaker.MakeCGCNNInput(atomic_numbers)
        if self.cuda:
            input = (input[0].cuda(non_blocking=True),#atom_fea
                     input[1].cuda(non_blocking=True),#nbr_fea
                     input[2].cuda(non_blocking=True),#nbr_atom_idx
                     [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])#crystal_atom_idx
        with torch.no_grad():
            output = self.model(*input)
        return self.normalizer.denorm(output.data.cpu()) * len(self.atoms)
           
