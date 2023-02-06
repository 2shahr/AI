import random
import statistics
import matplotlib.pyplot as plt

# random.seed(10)

rand = True


def initial(population_size, num_of_slots):
    chromosomes = []
    for i in range(population_size):
        chromosomes.append([random.randint(0, 1) for j in range(num_of_slots)])

    return chromosomes


def selection(chromosomes, num_of_slots):
    # print(chromosomes)
    fitness = []
    P = []
    PP = []
    E = []
    A = []
    for i in chromosomes:
        counter = 0
        f_tmp = 0.0
        for j in i:
            f_tmp += j*(2**counter)
            counter += 1
        fitness.append(f_tmp/num_of_slots)

    for i in fitness:
        P.append(i/sum(fitness))
        PP.append(i/sum(fitness)*100)
        E.append(i/statistics.mean(fitness))

    return fitness, P, PP, E


def roulette_wheel(wheel):
    m = sum(wheel.values())
    pick = random.uniform(0, m)
    current = 0
    for key, value in wheel.items():
        current += value
        if current > pick:
            return key


def crossover(chromosomes_dict, wheel, population_size, num_of_slots):
    chromosomes_dict_tmp = chromosomes_dict.copy()
    offspring = chromosomes_dict.copy()
    for i in range(population_size):
        chromosomes_dict[i] = chromosomes_dict_tmp[roulette_wheel(
            wheel)].copy()
    print(chromosomes_dict)
    for i in range(population_size):
        first_parent, second_parent, third_parent = random.sample(
            range(0, population_size), 3)
        # print(i)
        # print(first_parent, second_parent, third_parent)
        for j in range(num_of_slots):
            if (chromosomes_dict[first_parent][j] == chromosomes_dict[second_parent][j]):
                # print(j)
                # print(first_parent, second_parent, third_parent)
                # print(chromosomes_dict[first_parent][j])
                # print(chromosomes_dict[second_parent][j])
                # print('-----')
                offspring[i][j] = chromosomes_dict[first_parent][j]
            else:
                # print(j)
                # print(first_parent, second_parent, third_parent)
                # print(chromosomes_dict[first_parent][j])
                # print(chromosomes_dict[second_parent][j])
                # print('++++++')
                offspring[i][j] = chromosomes_dict[third_parent][j]
    # print(offspring)
    return offspring


def mutation(offspring, population_size, num_of_slots, rate):
    offspring_tmp = offspring.copy()
    for i in range(population_size):
        for j in range(num_of_slots):
            if random.random() < rate:
                offspring_tmp[i][j] = 1-offspring_tmp[i][j]
    return offspring_tmp


def main():
    sum_of_fitness = []
    population_size = 100
    num_of_slots = 8
    chromosomes_dict = {}
    rate = 0.13
    generation = 50
    if rand:
        chromosomes = initial(population_size, num_of_slots)
    else:
        chromosomes = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1], [
            0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1, 1]]

    for i in range(population_size):
        chromosomes_dict[i] = chromosomes[i]
    for i in range(generation):
        print('Generation: ', i)
        fitness, P, PP, E = selection(chromosomes_dict.values(), num_of_slots)
        print('Sum fitness', sum(fitness))
        sum_of_fitness.append(sum(fitness))
        print('Selection :')
        print([i for i in range(0, population_size)])
        print(fitness)
        print(P)
        print(PP)
        print(E)
        wheel = {}
        for i in range(population_size):
            wheel[i] = PP[i]
        print('generation: ', chromosomes_dict)
        print('Wheel: ', wheel)
        print('Crossover :')
        offspring = crossover(
            chromosomes_dict, wheel, population_size, num_of_slots)
        print('Next generation: ', offspring)
        mutaion_dict = mutation(offspring, population_size, num_of_slots, rate)
        print('Mutation: ', mutaion_dict)
        for i in range(population_size):
            chromosomes_dict[i] = mutaion_dict[i].copy()
        rate *= 0.7
        print('---------------------')

    plt.plot(sum_of_fitness)
    plt.show()


if __name__ == "__main__":
    main()
