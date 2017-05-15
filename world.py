#!/usr/bin/env python
from cosc343world import Creature, World
import numpy as np
import time
import random
import matplotlib.pyplot as plt

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 500

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the percept format.  You have three choice - format 1, 2 and 3 (described in
# the assignment 2 pdf document)
perceptFormat=1

# You can change this number to chnage the world size
gridSize=37

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generatin to generation.
repeatableMode=False

printStats = True

print ("Running world.py with: \nprintStats set to " + str(printStats) + "\nFor " + str(numGenerations) + " Generations")

stats_array = [[] for _ in range(12)]

# This is a class implementing you creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behavoiur by producing actions in respons to percepts.
class MyCreature(Creature):

    # Initialisation function.  This is where you creature
    # should be initialised with a chromosome in random state.  You need to decide the format of your
    # chromosome and the model that it's going to give rise to
    #
    # Input: numPercepts - the size of percepts list that creature will receive in each turn
    #        numActions - the size of actions list that creature must create on each turn
    def __init__(self, numPercepts, numActions):
        self.fitness = 0
        self.chromosome = np.random.random(8).tolist()
        self.mutate = 0.005
        '''
        Format
        -------------------------
        0. Monster move opposite
        1. Monster move closer
        2. Creature move opposite
        3. Creature move closer
        4. Food Move opposite
        5. Food Move closer
        6. Eat
        7. Random
        '''
        Creature.__init__(self)

    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned
    def AgentFunction(self, percepts, numActions):
        # Split percepts array into small sub arrays
        monster = percepts[:9]
        creature = percepts[9:18]
        food = percepts[18:]

        # Zero actions array
        actions = [0] * numActions

        # For each monster located multiply percept value by MMO and MMC chromosome values and add to the
        # actions array
        for percept_index, percept in enumerate(monster):
            if percept: # we found a non 0 percept
                for creature_index, monster_chromosome in enumerate(self.chromosome[:2]):
                    if creature_index == 0: # MMO
                        actions[8 - percept_index] += percept * monster_chromosome
                    else:
                        actions[percept_index] += percept * monster_chromosome

        # For each creature located multiply percept value by MMO and MMC chromosome values and add to the
        # actions array
        for percept_index, percept in enumerate(creature):
            if percept:
                for creature_index, creature_chromosome in enumerate(self.chromosome[2:4]):
                    if creature_index == 0:
                        actions[8 - percept_index] += percept * creature_chromosome
                    else:
                        actions[percept_index] += percept * creature_chromosome

        # For each food item located multiply percept value by MMA and MMC chromosome values and add to the
        # actions array
        for percept_index, percept in enumerate(food):
            if percept:
                for creature_index, food_chromosome in enumerate(self.chromosome[4:6]):
                    if creature_index == 0:
                        actions[8 - percept_index] += percept * food_chromosome
                    else:
                        actions[percept_index] += percept * food_chromosome

        # If we are sitting on food, increase chance to eat.        
        if food[4]:
            actions[9] += food[4] * self.chromosome[6]        
        # Count non-zero values and divide by total possible spots then add chromosome weight
        
        # map random action with chromosome value
        actions[10] = self.chromosome[7]
        actions[10] += (1 - (np.count_nonzero(percepts) / 27)) / 4

        return actions


# This function is called after every simulation, passing a list of the old population of creatures, whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last simulation.  You
#                         can query the state of the creatures by using some built-in methods as well as any methods
#                         you decide to add to MyCreature class.  The length of the list is the size of
#                         the population.  You need to generate a new population of the same size.  Creatures from
#                         old population can be used in the new population - simulation will reset them to starting
#                         state.
#
# Returns: a list of MyCreature objects of the same length as the old_population.
def newPopulation(old_population):
    global numTurns
    global stats_array
    global printStats
    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0

    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have a the end of simualation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    
    def get_fitness_score(individual):
        score = 0
        # Alive or dead.
        # Surivors with high energy get a legup over survivors with low energy
        if not individual.isDead():
            score += 50 + (individual.getEnergy() * 2)
        else:
            score += 25

        # Dead but lived for a long time
        if individual.timeOfDeath() > 50:
            score += individual.timeOfDeath() - 25

        # Dead, lived for a long time. Has high energy   
        if individual.timeOfDeath() > 50:
            score += individual.getEnergy()

        return score

    for individual in old_population:

        individual.fitness = get_fitness_score(individual)
        # This method tells you if the creature died during the simulation
        dead = individual.isDead()
        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath

        else:
            nSurvivors += 1
            avgLifeTime += numTurns
            ''' 
            # Uncomment to print chromosomes of survivors
            print("survived chromosomes")
            myRoundedList = [ round(elem, 2) for elem in individual.chromosome]
            print(myRoundedList)
            '''
        fitnessScore += individual.fitness
   
    # Stats
    avgLifeTime = float(avgLifeTime)/float(len(population))
    fitnessScore = float(fitnessScore)/float(len(population))
    
    stats_array[8].append(fitnessScore)
    stats_array[9].append(nSurvivors)
    stats_array[10].append(avgLifeTime)
    if printStats:
        print("Simulation stats:")
        print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
        print("  Avg  lifetime: %.1f turns" % avgLifeTime)
        print("  Avg Fitness  : %.1f" % fitnessScore)

    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals
    def tournament_selection(contenders):
        sorted_contenders = sorted(contenders, key=lambda x: x.fitness)
        return sorted_contenders[-1]

    def crossover(p1, p2):
        # Random cross over
        position = random.randint(0, 7)
        # Cross over
        child_chromosome = p1.chromosome[:position] + p2.chromosome[position:]

        child = MyCreature(numCreaturePercepts, numCreatureActions)
        child.chromosome = child_chromosome

        return child

    # Keep a number of elites to inject straight into the new population
    def elites(population, quantity_of_elites):
        elite_creatures = []
        # Sort population based on fitness
        sorted_population = sorted(population, key=lambda x: x.fitness)
        # 
        for i in range(len(sorted_population) - quantity_of_elites, len(sorted_population)):
            new_elite = MyCreature(numCreaturePercepts, numCreatureActions)
            new_elite.chromosome = sorted_population[i].chromosome
            elite_creatures.append(new_elite)

        return elite_creatures

    # Check if individual will mutate
    # value is stored in self.mutate
    def mutate(individual):
        if random.random() <= individual.mutate:
            individual.chromosome[random.randint(0, len(individual.chromosome)-1)] = random.random()
        return individual

    def prevent_incest(individual):
        individual.chromosome = np.random.random(8).tolist()

    new_population = elites(old_population, 2)
    sums = [0] * 8
    incest_count = 0
    incest_detected = 0
    while len(new_population) < len(old_population):
        # Tournament Select
        p1 = tournament_selection(random.sample(old_population, int(len(old_population)/5))) # 16
        p2 = tournament_selection(random.sample(old_population, int(len(old_population)/5)))

        # Count how many times chromosomes are the same
        if p1.chromosome == p2.chromosome:
            incest_count += 1
            incest_detected +=1

        if incest_count > 10:
        #     # Inject Randomness
            incest_count = 0
            prevent_incest(p2)


        # Cross over
        child = crossover(p1, p2)
        # Mutate
        child = mutate(child)

        for index, val in enumerate(child.chromosome):
            sums[index] += val

        new_population.append(child)
        # print(child.chromosome)

    stats_array[11].append(incest_detected)

    # Calculate averages and print
    averages = [0] * 8

    for index, val in enumerate(sums):
        averages[index] = val/len(new_population)


    for index, val in enumerate(averages):
        stats_array[index].append(val)

    if printStats:
        print ("Avg MMO: %.2f" % averages[0])
        print ("Avg MMC: %.2f" % averages[1])
        print ("Avg CMO: %.2f" % averages[2])
        print ("Avg CMC: %.2f" % averages[3])
        print ("Avg FMO: %.2f" % averages[4])
        print ("Avg FMC: %.2f" % averages[5])
        print ("Avg   F: %.2f" % averages[6])
        print ("Avg   R: %.2f" % averages[7])

        print ("5 Random Chromosomes")
        # print 5 random chromosomes from within the new_population
        for i in range(5):
            i = random.randint(0, len(new_population)-1)
            myRoundedList = [ round(elem, 2) for elem in new_population[i].chromosome]
            print(myRoundedList)

    return new_population

# Create the world.  Representaiton type choses the type of percept representation (there are three types to chose from);
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation in exactly same way.
w = World(representationType=perceptFormat, gridSize=gridSize, repeatable=repeatableMode)

#Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

#Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

#Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
   c = MyCreature(numCreaturePercepts, numCreatureActions)
   population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evalute the first population
w.evaluate(numTurns)

# Show visualisation of initial creature behaviour
#w.show_simulation(titleStr='Initial population', speed='fast')

for i in range(numGenerations):
    if printStats:
        print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    if i==numGenerations-1:
        w.show_simulation(titleStr='Final population', speed='normal')

#plt.plot(stats_array[0], color='blue', label="MMA")
#plt.plot(stats_array[1], color='red', label="MMC")
plt.plot(stats_array[9], color='blue', label="Survivors")
#plt.plot(stats_array[11], color='green', label="Incest Detected")
#plt.plot(stats_array[10], color='red', label="Avg Turns Survived")
plt.plot(stats_array[8], color='purple', label="Fitness")
#plt.title("Average Fitness of Generation and Survivors")
plt.title("Average Fitness and Number of Survivors in Each Generation")

plt.xlabel("Number of Generations")
#plt.ylabel("Average Fitness")
plt.legend(loc='lower right')
plt.show()
