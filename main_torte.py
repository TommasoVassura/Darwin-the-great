# Esame di Algoritmi Genetici

import random
from numpy import NaN
import argparse

NTORTE = 10
ENERGIA = 10
NGRIGLIA = 10
POSSIBILITA = [ format(i, "04b") for i in range(0,16)]

class Creature():

    def __init__(self, dizionario = None, x=None, y=None, energia = ENERGIA  ):
        ''' se non do niente genera delle mosse e delle posizioni casualmente con energia quella iniziale (e anche massima) '''
        if dizionario: #gli ho dato qualche dizionario
            self.mosse = dizionario
        else: #genero casualmente
            self.mosse = {i : random.randint(0, 4) for i in POSSIBILITA} #0 è destra, 1 su, 2 sinistra, 3 giù

        if x==None & y==None: 
            self.x=random.randint(0,NGRIGLIA)
            self.y=random.randint(0,NGRIGLIA)
        else:
            self.x=x
            self.y=y
        
        self.energia = energia

class Ambiente():

    pass

# PROGRAMMA PRINCIPALE

if __name__=='__main__':

    parser = argparse.ArgumentParser(description = "A program running genetic algorithms")
    parser.add_argument('ncities', help='number of cities', type=int)
    parser.add_argument('popsize', help= "Population size", type=int)
    parser.add_argument('surv_frac', help= "Surviving fraction", type=float)
    parser.add_argument('mut_prob', help= "Mutation probability", type=float)
    parser.add_argument('ngen', help= "N of generations", type=int)
    parser.add_argument('--no_plots', help="does not show plots", default=False, action='store_true')
    args = parser.parse_args()

    print(args)

    npop = args.popsize
    surv_prob = args.surv_frac
    mut_prob = args.mut_prob
    ngen = args.ngen
    ncities = args.ncities

        