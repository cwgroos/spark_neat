from __future__ import print_function

import neat
import os
import random

from pyspark import SparkContext

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

sc = SparkContext(appName="Neat-Eval")

def xor_func(id, genome, config, inputs):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    output = net.activate(inputs)
    #error = (output[0] - outputs[0]) ** 2
    return (id, output)

def eval_genomes(genomes, config):
    try:
        id2genome = {genome_id: genome for genome_id, genome in genomes} 
        for key in id2genome:
            id2genome[key].fitness = 4.0
        rrd_genomes = sc.parallelize(genomes)
        for xi, xo in zip(xor_inputs, xor_outputs):
            errors = rrd_genomes.map(lambda g: xor_func(g[0], g[1], config, xi)).collect()
            for id, error in errors:
                id2genome[id].fitness -= (error[0] - xo[0]) ** 2
    except KeyboardInterrupt:
        exit()

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.getcwd(), 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    pop = neat.Population(config)
    best_genome = pop.run(eval_genomes, 500)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config) 
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    sc.stop()

if __name__ == "__main__":
        run()