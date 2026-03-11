import random
# Parameters
POPULATION_SIZE = 1000
GENES = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890'
TARGET = "I love MSCDSAI with RJColleges"
class Individual:
    """Represents a single solution (chromosome)"""
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calc_fitness()
    @classmethod
    def random_gene(cls):
        return random.choice(GENES)
    @classmethod
    def create_random(cls):
        return [cls.random_gene() for _ in range(len(TARGET))]
    def mate(self, partner):
        """Crossover + mutation to produce child"""
        child_chromosome = [
            (self.random_gene() if random.random() > 0.9 else
             g1 if random.random() < 0.5 else g2)
            for g1, g2 in zip(self.chromosome, partner.chromosome)
        ]
        return Individual(child_chromosome)
    def calc_fitness(self):
        """Fitness = number of mismatched characters"""
        return sum(1 for gs, gt in zip(self.chromosome, TARGET) if gs != gt)
def main():
    generation = 1
    found = False
    # Initial population
    population = [Individual(Individual.create_random()) for _ in range(POPULATION_SIZE)]
    while not found:
        # Sort population by fitness
        population.sort(key=lambda ind: ind.fitness)
        # Check if we reached the target
        best = population[0]
        print(f"Generation {generation} | String: {''.join(best.chromosome)} | Fitness: {best.fitness}")
        if best.fitness == 0:
            found = True
            break
        # Elitism: keep top 10%
        new_gen = population[:POPULATION_SIZE // 10]
        # Generate remaining 90% via crossover
        for _ in range(POPULATION_SIZE - len(new_gen)):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            new_gen.append(parent1.mate(parent2))
        population = new_gen
        generation += 1
    print(f"\nFinal Generation: {generation}\nString: {''.join(best.chromosome)}\nFitness: {best.fitness}")
if __name__ == "__main__":
    main()
